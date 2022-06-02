# 参数设置

| 参数名称      | 参数类型 | 参数描述                                           |
| ------------- | -------- | -------------------------------------------------- |
| target        | str      | 预测目标                                           |
| multi_step    | bool     | 单步预测或多步预测                                 |
| seq_length    | int      | 时间步长划分                                       |
| batch_size    | int      |                                                    |
| feature       | list     | 特征                                               |
| mode          | str      | 训练或测试（['train', 'test']）                    |
| input_size    | int      | 输入数据的大小，和特征数一致即可                   |
| hidden_size   | int      | 隐藏层神经元个数                                   |
| num_layers    | int      | lstm层数                                           |
| output_size   | int      | 输出的size，如果单步设置为1                        |
| bidirectional | bool     | 是否为双向lstm                                     |
| qkv           | int      | 注意力机制中qkv的纬度                              |
| which_data    | str      | 数据集路径（csv）                                  |
| weights_path  | str      | pkl文件存储路径                                    |
| plots_path    | str      | 结果保存路径                                       |
| train_split   | float    | 训练集划分比例                                     |
| test_split    | float    | 测试集合划分比例（训练集剩下的划分验证集和测试集） |
| time_plot     | int      | 结果图上展示的前多少个时刻的数据                   |
| num_epochs    | int      |                                                    |
| print_every   | int      | 训练过程展示                                       |
| lr            | float    | 学习率                                             |
| model         | str      | 模型类型，['cnn', 'rnn', 'lstm', 'attentional']    |

项目中有一部分已经训练好的pkl文件，调用test，设置好相应的参数即可。

如BEST_AttentionalLSTM_using_MultiStep_Window_8_Before_2.pkl

其中AttentionalLSTM为模型名称，由model参数设置，8为时间步长，由seq_length设置，2为提前预测的时间步，由output_size配合multi_step（true）设置。
