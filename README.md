# Dynamic Attention Model for Vehicle Routing Problems

### DeepPavlov course: Advanced Topics in Deep Reinforcement learning (http://deeppavlov.ai/rl_course_2020)

### Dmitry Eremeev, Alexey Pustynnikov

One of the most important applications of combinatorial optimization is vehicle routing problem, in
which the goal is to find the best routes for a fleet of vehicles visiting a set of
locations. Usually, "best" means routes with the least total distance or cost.

We would consider only particular case of general VRP problem: Capacitated
Vehicle Routing Problem (CVRP), where the vehicle has a limited carrying
capacity of the goods that must be delivered.

<img src="pictures/VRP.jpg" style="width: 20px;"/>

VRP is an NP-hard problem (Lenstra and Rinnooy Kan, 1981). 

Exact algorithms are only efficient for small problem instances.
The number of near-optimal algorithms are introduced in academic
literature. There are multiple professional tools for solving various VRP
problems (ex. Google OR-Tools).

### CVRP: Graph Formulation

- Complete graph:

$$X=(V,E),$$

  with set of nodes

$$V=\left\{x_0 \equiv D,x_1,x_2,\ldots,x_n \right\}.$$


- Each node is associated with a demand $d_i$, $d_i < C$, $d_D = 0$.

- Each edge is associated with a cost $c_{ij}$ ($L^2$-norm).

- A vehicle with capacity $C$ is moving along graph starting from depot node $D$. Every non-depot node $x_k$ can be visited only once. It is allowed to return to $D$ arbitrary many times.

- Goal: find a path $\pi=\left\{\pi_1, \ldots,  \pi_T\right\}$, $\pi_t \in V$ that minimizes **total cost**.
### Attention Model Aproach

The structural features of the input graph instance are extracted by the encoder.
Then the solution is constructed incrementally by the decoder. 

Specifically, at each construction step, the decoder predicts a distribution over nodes, then one
node is selected and appended to the end of the partial solution. Hence,
corresponding to the parameters $\theta$ and input instance $X$, the probability of
solution $p_\theta(\pi|X)$ can be decomposed by chain rule as:

$$p_\theta(\pi|X) = \prod_{t=1}^{T}p_\theta(\pi_t|X,\pi_{1:t-1})$$ 

where $T$ is the length of solution. 

#### Main ideas:

- Use RL to create agent that can learn heuristics and provide suboptimal solutions.
- Make use of Graph Attention Networks (GAT) to create appropriate graph embeddings for the agent.

#### Architecture:

<img src="pictures/nn_architecture_cdr_v2.png" alt="Drawing" style="width: 400px;"/>


#### Dynamic Attention Model (AM-D) Approach:

- After vehicle returns to depot, the remaining nodes could be considered as a new (smaller) instance (graph) to be solved.
- <font color='green'><b>Idea: update embedding of the remaining nodes using encoder after agent arrives back to depot. </b></font>

- Implementation:

 1) Force RL agent to wait for others once it arrives to $x_0$.
 
 2) When all are in depots, apply encoder with mask to the whole batch.
 
 3) Typical solution will be of the form: [17., 3., 4., 7., 2., 16., 0., 0., 0., 0., 15., 20., 5.,
0., 0., 0., 0., 11., 12., 13., 10., 9., 0., 0., 0., 19.,
6., 8., 14., 1., 18., 0.]


#### Enviroment:

Current enviroment implementation is located in **enviroment.py** file -- <font color='darkorange'> AgentVRP class </font>.

The class contains information about current state and actions that were done by agent.

Main methods:

- **step(action)**: transit to new state according to the action.
- **get_costs(dataset, pi)**: returns costs for each graph in batch according to the paths in action-state space.
- **get_mask()**: returns a mask with available actions (allowed nodes).
- **all_finished()**: checks if all games in batch are finished (all graphes are solved).
- **partial_finished()**: checks if partial solutions for all graphs has been built, i.e. all agents came back to depot.

*Example*: 

Suppose we have a graph in initial state (*state = AgentVRP(graph_instance)*), i.e. the agent is in the depot (node $x_0$). Then *state.from_depot* is *True*. If the goal is to send the agent to node five, then one can write *state.step([4])* and it's done.

Let's connect current terms with RL language (small dictionary):

- **State**: $X$ - graph instance (coordinates, demands, etc.) together with information in which node agent is located.
- **Action**: $\pi_t$ - decision in which node agent should go.
- **Reward**: The (negative) tour length.

#### Model Training:

AM-D is trained by policy gradient using <font color='navy'><b>REINFORCE</b></font> algorithm.

$$\nabla_{\theta} J(\theta) \sim \mathbb{E}_p\left[(L^p(X,\pi)-b(X))\nabla_{\theta} \log(p_\theta(\pi|X))\right],$$

where conditional probability of solution is:

$$p_\theta(\pi|X) = \prod_{t=1}^{T}p_\theta(\pi_t|X,\pi_{1:t-1}).$$ 


**Baseline**

- Baseline is a <font color='navy'><b>copy of model</b></font> with fixed weights from one of the preceding epochs.
- Use warm-up for early epochs: mix exponential moving average (controlled by $\beta=\text{const}$) of model cost over past epochs with baseline model. Warm-up is controlled by $\alpha \in [0; 1]$.
- Update baseline at the end of epoch if the difference in costs for candidate model and baseline is statistically-significant (t-test).
- Baseline uses separate dataset for this validation. This dataset is updated after each baseline renewal.

**Training**

- Estimate model cost by Monte Carlo: generate an episode $S_1, A_1, \ldots, S_{T}, A_{T}$, following $p_\theta(\cdot | \cdot)$ in sampling mode (stochastic policy). Then loop through all steps to get cost of the whole episode.
- Evaluate baseline in greedy mode (selects the node with maximum probability - deterministic policy).
- Estimate gradient according to policy-gradient formula and update weights of the neural network.

**Example**

<img src="pictures/tour_dynamics.gif" alt="Drawing" style="width: 500px;"/>

# Files Description:

Implementation in **TensorFlow 2**

 0) **AM-D for VRP Report.ipynb** - demo report notebook
 1) **enviroment.py** - enviroment for VRP RL Agent
 2) **attention_graph_encoder.py** - Graph Attention Encoder
 3) **layers.py** - MHA layers for encoder
 4) **reinforce_baseline.py** - class for REINFORCE baseline
 5) **attention_dynamic_model.py** - main model and decoder
 6) **train.py** - defines training loop, that we use in train_model.ipynb
 7) **train_model.ipynb** - from this file one can start training or continue training from chechpoint
 8) **utils.py** and **utils_demo.py** - various auxiliary functions for data creation, saving and visualisation
 9) **lkh3_baseline** folder - everything for running LKH algorithm + logs.
 10) results folder: 
                    1) folder name ADM_VRP_{graph_size}_{batch_size}
                    2) there are training logs, learning curves and saved models in each folder 
 
 # Training procedure:
 
  1) For training on large batches (> 128) it may require more than 8 Gb of GPU memory
  2) Open  **train_model.ipynb** and choose training parameters
  3) All outputs would be saved in current directory
