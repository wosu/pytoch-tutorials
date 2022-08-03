import os
from functools import partial

from ray import tune
import numpy as np
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
from torch import nn, optim


def init_model(config):
    pass


def load_data(data_dir):
    pass


def model_train(config, checkpoint_dir=None, data_dir=None, num_epoch=10):
    # 从config获取超参init model
    model = init_model(config)
    # 并行化
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # 定义loss
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    # checkpoint
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    # 加载训练和测试数据
    train_loader, test_loader = load_data(data_dir)
    # 迭代训练
    for epoch in range(num_epoch):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            # forward
            outputs = model(inputs)
            # 计算loss
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 计算梯度
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        # checkpoint
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        # 回调指标，需要监控的指标
        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)


# 模型需要搜索的超参数
config = {
    "layer_1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),  # 第一层layer的神经元数量
    "layer_2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),  # 第二层layer的神经元数量
    "lr": tune.loguniform(1e-4, 1e-1),  # 学习率
    "batch_size": tune.choice([2, 4, 8, 16])  # batch_size
}
# 定义ray的scheduler  Scheduler for executing the experiment
'''
metric="loss" 这里的metric对应训练过程中最后ray.reporter中的指标
mode="min" 表示metric目标，这里是最小化loss
'''
max_num_epochs = 10
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2)

reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"])
data_dir = ""
'''
resources_per_trial:  Note that GPUs will not be
            assigned unless you specify them here. Defaults to 1 CPU and 0
            GPUs i
num_samples (int): Number of times to sample from the
            hyperparameter space            
'''
gpus_per_trial = 0
num_samples = 10
# 开始执行
result = tune.run(
    partial(model_train, data_dir=data_dir),
    resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter)

# 返回结果
best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))