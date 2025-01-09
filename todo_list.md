- ~~测试MessageBus的功能~~
- ~~增加Reducer的功能~~
- ~~TensorRegistryManager的功能由SimulatorContext接管~~
- ~~编写GeometryManager~~
- ~~配置。。如何告诉仿真当前的环境中对应的机器人数量？在register。增加onConfigure回调，并在register时dosomething~~
- ~~Component如何在运行时获得环境组的参数？ExecuteContext：当前活动的环境组id。当前Kernel~~
- ~~constant memory 读取接口, sync_to_device~~
- 移植地图生成、Gridmap
- 修改message queue创建shapes时的batch
- 什么时候sync_data?

- python接口
- Node的onReset接口
- TensorHandle* 改为TensorHandle

- ReducerComponent改名

- 环境组在Kernel中的访问接口
- 环境组数量，环境组生成的逻辑，环境组随机化的相关接口，运行配置？
- component与环境组之间的接口，与随机化器之间的接口？
- ConfigurableObject类，以支持Config配置

- 移植并测试激光雷达
- 移植一个brax引擎测试
- 移植原有初始pbd版本引擎
- 移植Geniue引擎的相关实现
- onRegister -> onNodeRegister
- 环境组随机化时是否需要配置文件？