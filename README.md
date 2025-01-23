# RCS NextGen

This repository contains Robot CUDA Simulator. Its architecture is inspired by Robot Operating System(ROS), which include a pub-sub design for flexible component design.

## Motivation

A large body of game theory research has utilized simulation environments for planar robots, including many studies based on reinforcement learning methods. These algorithms typically require a significant amount of sampling to converge to Nash equilibria, and the sampling speed is one of the key factors affecting the efficiency of these algorithms.


In recent years, some studies have been working to alleviate the sampling problem in reinforcement learning by using hardware-accelerated simulations, such as Isaac Gym. However, the game theory community has rarely directly benefited from these advancements. We believe the primary reason lies in the differences in task objectives between the two fields. Most existing studies focus on control tasks, such as robotic arms or humanoid robots. Their main work involves parallelizing physical simulations (e.g., differentiable physics). On the other hand, game theory requires customization and adjustments within specific environment configurations and rule settings, necessitating simulation environments with a high degree of flexibility. Most existing simulators are structured as monolithic, single-file systems, making it difficult for users to perform secondary development, such as adding new CUDA-accelerated functional components.


We address this issue by developing a new simulation framework that supports GPU acceleration. First, inspired by the ROS system, we designed a publish-subscribe message bus to decouple various simulation components, such as physics simulation, sensors, and controllers. Each component declares its input and output data shapes during initialization, and the message bus constructs a static computation graph based on the dependencies between messages. The advantage of this design is that users can freely modify simulation components or even integrate external tools such as Brax into the simulation. Additionally, the publish-subscribe mechanism enhances code reusability.


Second, we introduced the concept of environment groups to support domain randomization. An environment group consists of a batch of environments with identical simulation parameters (e.g., maps, friction coefficients). The parameters of each environment group are sampled from a distribution defined by domain randomization. During execution, a batch is sampled from all environment groups for simultaneous execution, which is designed to facilitate sim-to-real transfer in game-theoretic algorithms.

## Getting Started

This project has not been finished yet. However, the core function such as Message Bus, Environment Group, and Python Binding is availible now. We are now developing new components such as simple physics engine and so on.

## Contributing

We welcome contributions! Please submit a pull request as you wish.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

如果这个仓库对你有帮助，请你帮忙点一个STAR吧~
如果你有什么问题，欢迎提出ISSUE，我会尽力帮助你解决问题。
