# Visuomotor Diffusion Policies TRI

## About
Notes on Visuomotor Diffusion Policies TRI 

## (Visuomotor) Diffusion Policies TRI 

### Video of Tedrake

[Princeton Robotics - Russ Tedrake - Dexterous Manipulation with Diffusion Policies](https://www.youtube.com/watch?v=whpK0HDtOJ0&t=39s)

Inputs: cameras images and robots states

- Image backbone: ResNet-18 (pretrained on image net)
- Total: 110M - 150M params
- Training time: 3-6 GPU Days ($150-$300) to train a new skill with about 50 demonstrations
- OPeration Control: low gain stiffness control. With graity compensation. Robot is complient. The positions are sent.
- Multitask: Take data from many skills and transform them to use as language conditional policies
- Core challenges:
    - Control from pixels
    - Control through contact
    - Optimizing rich robustness objective
    - Deep RL + Teacher-Student
- GSC (graphs of convex sets)

Other methods:
- Score functions
- Action chunking transformer (ACT)
- Categorical distribution (RT1, RT2)

#### Other methods

- [Policy Composition From and For Heterogeneous Robot Learning](https://liruiw.github.io/policycomp/)
- Latent diffusion (based on image embeddings instead of images directly)
- cross attention
- text conditioning
- classifier-free guidance


### Paper

[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)

Diffusion Policy learns the gradient of the action-distribution score function and iteratively optimizes with respect to this gradient field during inference via a series of stochastic Langevin dynamics steps. 

It handles multimodal action distributions, being suitable for high-dimensional action spaces, and exhibiting impressive training stability.

Key technical contributions:
- the incorporation of receding horizon control
- visual conditioning
- the time-series diffusion transformer. 

Code, data, and training details will be publicly available.

Policy learning from demonstration can be formulated as the supervised regression task of learning to map observations to actions.
However challenging on real robots, because of multimodal distributions, sequential correlation, and the requirement of high precision.

The paper addresses this challenge by introducing a new form of robot visuomotor policy that generates behavior via a “conditional denoising diffusion process [abbel et al] on robot action space”, Diffusion Policy. In this formulation, instead of directly outputting an action, the policy infers the action-score gradient, conditioned on visual observations, for K denoising iterations. 

Properties from diffusion models:
- Expressing multimodal action distributions.By learning the gradient of the action score function [46] and performing Stochastic Langevin Dynamics sampling on this gradient field,
- High-dimensional output space ( This property allows the policy to jointly infer a sequence of future actions instead of single-step actions, which is critical for encouraging temporal action consistency and avoiding myopic planning.)
- Stable training. by learning the gradient of the energy function

Technical contributions of the paper:
- Closed-loop action sequences.
- Visual conditioning. The visual observations are treated as conditioning instead of a part of the joint data distribution, extracting the visual representation once, reducing the computation and enabling real-time action inference.
- Time-series diffusion transformer. 

Evaluation was done across 12 tasks from 4 different benchmarks. The evaluation includes both simulated and real-world environments, 2DoF to 6DoF actions, single- and multitask benchmarks, and fully- and under-actuated systems, with rigid and fluid objects, using demonstration data collected by single and multiple users. With an average improvement of 46.9% to existing state-of-the-art robot learning methods.

We formulate visuomotor robot policies as Denoising Diffusion Probabilistic Models (DDPMs).

DDPMs are a class of generative model where the output generation is modeled as a denoising process, often called Stochastic Langevin Dynamics.

The DDPM is used to learn robot visuomotor policies. This requires two major modifications in the formulation: 
1. changing the output x to represent robot actions. See paper [30]. at time step t the policy takes the latest To steps of observation data Ot as input and predicts Tp steps of actions, of which Ta steps of actions are executed on the robot without re-planning. Here, we define To as the observation horizon, Tp as the action prediction horizon and Ta as the action execution horizon. 
2. making the denoising processes conditioned on input observation Ot. DDPM is used to approximate the conditional distribution p(At |Ot ) instead of the joint distribution p(At , Ot ) used in [20], allowing the model to predict actions conditioned on observations without the cost of inferring future states.

Two network architecture types, convolutional neural networks (CNNs) and Transformers were used to estimate εθ - the noise prediction network with parameters θ that will be optimized.
CNN-based Diffusion Policy as of [21] were used with some adoptation. CNN-based backbone works well without the need for much hyperparameter tuning. It performs poorly when the desired action sequence changes quickly and sharply through time.
For transformer-based DDPM, the transformer architecture from minGPT was adopted. Actions with noise Akt are passed in as input tokens for the transformer decoder blocks, with the sinusoidal embedding for diffusion iteration k prepended as the first token. The observation Ot is transformed into observation embedding sequence by a shared MLP, which is then passed into the transformer decoder stack as input features. The "gradient" εθ (Ot, Atk, k) is predicted by each corresponding output token of the decoder stack.

Visual Encoder maps the raw image sequence into a latent embedding Ot and is trained end-to-end with the diffusion policy. A standard ResNet-18 (without pretraining) were used as the encoder with the following modifications: 1) Replace the global average pooling with a spatial softmax pooling to maintain spatial information [29]. 2) Replace BatchNorm with GroupNorm [57] for stable training. 

Square Cosine Schedule proposed in iDDPM [33] was used as a noise Scheduler.

Denoising Diffusion Implicit Models was used to accelerate inference. 100 training iterations and 10 inference iterations enables 0.1s
inference latency on an Nvidia 3080 GPU.

Diffusion Policy’s ability to express multimodal distributions naturally and precisely is one of its key advantages. multi-modality in action generation for diffusion policy arises from two sources – an underlying stochastic sampling procedure and a stochastic initialization.

We find that Diffusion Policy with a position-control action space consistently outperforms Diffusion Policy with velocity control.

It is more stable on training as other methods because of the score function.

Key Findings

Diffusion Policy can express short-horizon multimodality. We define short-horizon action multimodality as multiple ways of achieving the same immediate goal, which is prevalent in human demonstration data.

Diffusion Policy can express long-horizon multimodality. Long-horizon multimodality is the completion of different sub-goals in inconsistent order.

Diffusion Policy can better leverage position control. Our ablation study (Fig. 5) shows that selecting position control as the diffusion-policy action space significantly outperformed velocity control.

The tradeoff in action horizon. As discussed in Sec IV-C, having an action horizon greater than 1 helps the policy predict consistent actions and compensate for idle portions of the demonstration, but too long a horizon reduces performance due to slow reaction time. action horizon of 8 steps to be optimal. 

Robustness against latency. Diffusion Policy employs receding horizon position control to predict a sequence of actions into the future. This design helps address the latency gap caused by image processing, policy inference, and network delay. peak performance with latency up to 4 steps.

Diffusion Policy is stable to train.

REALWORLD EVALUATION

Diffusion Policy predicts robot commands at 10 Hz and these commands then linearly interpolated to 125 Hz for robot execution.


## Questions

- multi-modal action distribution?
- which existing state-of-the-art robot learning methods were used by evaluation?
- what are the observation features Ot exactly?
- what are the Aaction Sequence At exactly?


## Evaluations

[Robomimic: A Framework for Robot Learning from Demonstration](https://robomimic.github.io/)

[The COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation](https://robot-colosseum.github.io/)

[Franka Kitchen](https://robotics.farama.org/envs/franka_kitchen/franka_kitchen/)

## Links

### ToRead
[30] David Q Mayne and Hannah Michalska. Receding horizon control of nonlinear systems. In Proceedings of the 27th IEEE Conference on Decision and Control, pages 464–465. IEEE, 1988.

[20] Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, 2022. 3, 11

[21] Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, Proceedings of Machine Learning Research. PMLR, 17–23 Jul 2022. 3

[29] Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. In 5th Annual Conference on Robot Learning, 2021. 1, 2, 4, 6, 7, 10, 14, 15

[12] Pete Florence, Corey Lynch, Andy Zeng, Oscar A Ramirez, Ayzaan Wahid, Laura Downs, Adrian Wong, Johnny Lee, Igor Mordatch, and Jonathan Tompson. Implicit behavioral cloning. In 5th Annual Conference on Robot Learning, 2021. 1, 2, 4, 5, 6, 7, 8, 10

### Diffusion models foundation

[miniDiffusion](https://github.com/cloneofsimo/minDiffusion)
[diffusion-models - short course on deeplearning.ai](https://learn.deeplearning.ai/diffusion-models/)

Diffusion model is based on [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

### Diffusion score function

[Score matching and Langevin dynamics, Diffusion probabilistic models and denoising, Github with Jupyter notebooks](https://github.com/acids-ircam/diffusion_models/tree/main)
[Diffusion and Score-Based Generative Models, video](https://www.youtube.com/watch?v=wMmqCMwuM2Q)
[Generative Modeling by Estimating Gradients of the Data Distribution, Blog with Colabs](https://yang-song.net/blog/2021/score/)

### Multimodality

[Multimodal Deep Learning Book](https://arxiv.org/abs/2301.04856)
[Multimodal Learning with Transformers: A Survey](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjRq-7blbmEAxUWcPEDHV7KA-wQFnoECBYQAQ&url=https%3A%2F%2Farxiv.org%2Fpdf%2F2206.06488&usg=AOvVaw0TXfdSTMbgNnQVKZryT65G&opi=89978449)

### Robot pose representation in 3D  by learning
[On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035)