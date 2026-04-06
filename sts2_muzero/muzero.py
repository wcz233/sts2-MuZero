import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    maximum = max(logits)
    exponents = [math.exp(value - maximum) for value in logits]
    total = sum(exponents)
    if total <= 0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exponents]


def _masked_softmax(logits: list[float], legal_actions: list[int]) -> list[float]:
    probabilities = [0.0] * len(logits)
    if not legal_actions:
        return probabilities
    legal_probabilities = _softmax([logits[action] for action in legal_actions])
    for action, probability in zip(legal_actions, legal_probabilities):
        probabilities[action] = probability
    return probabilities


def _clamp(value: float, bound: float = 5.0) -> float:
    if value < -bound:
        return -bound
    if value > bound:
        return bound
    return value


@dataclass
class Transition:
    observation: list[float]
    action_index: int
    reward: float
    next_observation: list[float]
    done: bool
    policy_target: list[float]


@dataclass
class SearchResult:
    action_probabilities: list[float]
    visit_counts: list[int]
    root_value: float


@dataclass
class TrainMetrics:
    total_loss: float = 0.0
    value_loss: float = 0.0
    reward_loss: float = 0.0
    policy_loss: float = 0.0
    consistency_loss: float = 0.0


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[Transition] = []
        self.position = 0

    def append(self, transition: Transition) -> None:
        if len(self.items) < self.capacity:
            self.items.append(transition)
        else:
            self.items[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, rng: random.Random) -> Transition:
        return self.items[rng.randrange(len(self.items))]


class LinearLayer:
    def __init__(self, rng: random.Random, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size
        scale = 1.0 / math.sqrt(max(1, input_size))
        self.weights = [[rng.uniform(-scale, scale) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [0.0 for _ in range(output_size)]

    def forward(self, inputs: list[float]) -> list[float]:
        return [_dot(row, inputs) + bias for row, bias in zip(self.weights, self.biases)]

    def backward(self, inputs: list[float], grad_outputs: list[float], learning_rate: float) -> list[float]:
        grad_inputs = [0.0 for _ in range(self.input_size)]
        for row_index, raw_grad in enumerate(grad_outputs):
            grad = _clamp(raw_grad)
            if grad == 0.0:
                continue
            row = self.weights[row_index]
            for input_index, weight in enumerate(row):
                grad_inputs[input_index] += weight * grad
            for input_index, input_value in enumerate(inputs):
                row[input_index] -= learning_rate * grad * input_value
            self.biases[row_index] -= learning_rate * grad
        return grad_inputs

    def state_dict(self) -> dict[str, object]:
        return {"weights": self.weights, "biases": self.biases}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.weights = [[float(value) for value in row] for row in state["weights"]]  # type: ignore[index]
        self.biases = [float(value) for value in state["biases"]]  # type: ignore[index]


class TanhLinearLayer(LinearLayer):
    def forward(self, inputs: list[float]) -> tuple[list[float], list[float]]:
        outputs = [math.tanh(value) for value in super().forward(inputs)]
        return outputs, outputs

    def backward(self, inputs: list[float], outputs: list[float], grad_outputs: list[float], learning_rate: float) -> list[float]:
        grad_preactivations = [_clamp(grad * (1.0 - output * output)) for grad, output in zip(grad_outputs, outputs)]
        return super().backward(inputs, grad_preactivations, learning_rate)


class MuZeroNetwork:
    def __init__(self, observation_size: int, action_size: int, hidden_size: int, seed: int) -> None:
        rng = random.Random(seed)
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.representation = TanhLinearLayer(rng, observation_size, hidden_size)
        self.dynamics = TanhLinearLayer(rng, hidden_size + action_size, hidden_size)
        self.reward_head = LinearLayer(rng, hidden_size, 1)
        self.policy_head = LinearLayer(rng, hidden_size, action_size)
        self.value_head = LinearLayer(rng, hidden_size, 1)

    def represent(self, observation: list[float], cache: bool = False) -> tuple[list[float], dict[str, object] | None]:
        hidden, outputs = self.representation.forward(observation)
        return hidden, {"observation": observation, "outputs": outputs} if cache else None

    def predict(self, hidden: list[float]) -> tuple[list[float], float]:
        return self.policy_head.forward(hidden), self.value_head.forward(hidden)[0]

    def transition(self, hidden: list[float], action_index: int, cache: bool = False) -> tuple[list[float], float, dict[str, object] | None]:
        action_vector = [0.0 for _ in range(self.action_size)]
        action_vector[action_index] = 1.0
        dynamics_input = hidden + action_vector
        next_hidden, outputs = self.dynamics.forward(dynamics_input)
        reward = self.reward_head.forward(next_hidden)[0]
        if not cache:
            return next_hidden, reward, None
        return next_hidden, reward, {"dynamics_input": dynamics_input, "outputs": outputs, "next_hidden": next_hidden}

    def backward_prediction(self, hidden: list[float], grad_logits: list[float], grad_value: float, learning_rate: float) -> list[float]:
        grad_from_policy = self.policy_head.backward(hidden, grad_logits, learning_rate)
        grad_from_value = self.value_head.backward(hidden, [grad_value], learning_rate)
        return [left + right for left, right in zip(grad_from_policy, grad_from_value)]

    def backward_transition(self, cache: dict[str, object], grad_next_hidden: list[float], grad_reward: float, learning_rate: float) -> list[float]:
        next_hidden = cache["next_hidden"]  # type: ignore[index]
        dynamics_input = cache["dynamics_input"]  # type: ignore[index]
        outputs = cache["outputs"]  # type: ignore[index]
        grad_from_reward = self.reward_head.backward(next_hidden, [grad_reward], learning_rate)
        total_grad = [left + right for left, right in zip(grad_next_hidden, grad_from_reward)]
        grad_inputs = self.dynamics.backward(dynamics_input, outputs, total_grad, learning_rate)
        return grad_inputs[: self.hidden_size]

    def backward_representation(self, cache: dict[str, object], grad_hidden: list[float], learning_rate: float) -> None:
        self.representation.backward(cache["observation"], cache["outputs"], grad_hidden, learning_rate)  # type: ignore[index]

    def state_dict(self) -> dict[str, object]:
        return {
            "representation": self.representation.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.representation.load_state_dict(state["representation"])  # type: ignore[index]
        self.dynamics.load_state_dict(state["dynamics"])  # type: ignore[index]
        self.reward_head.load_state_dict(state["reward_head"])  # type: ignore[index]
        self.policy_head.load_state_dict(state["policy_head"])  # type: ignore[index]
        self.value_head.load_state_dict(state["value_head"])  # type: ignore[index]


class MuZeroAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.01,
        discount: float = 0.997,
        simulations: int = 24,
        replay_capacity: int = 4096,
        warmup_samples: int = 32,
        seed: int = 7,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
    ) -> None:
        self.network = MuZeroNetwork(observation_size, action_size, hidden_size, seed)
        self.learning_rate = learning_rate
        self.discount = discount
        self.simulations = simulations
        self.warmup_samples = warmup_samples
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.replay = ReplayBuffer(replay_capacity)
        self.rng = random.Random(seed + 17)
        self.training_steps = 0

    def plan(self, observation: list[float], legal_actions: list[int], use_exploration_noise: bool = True) -> SearchResult:
        hidden, _ = self.network.represent(observation, cache=False)
        logits, root_value = self.network.predict(hidden)
        priors = _masked_softmax(logits, legal_actions)
        if use_exploration_noise and len(legal_actions) > 1:
            noise = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in legal_actions]
            total_noise = sum(noise)
            if total_noise > 0:
                for action, sample in zip(legal_actions, noise):
                    priors[action] = (1.0 - self.exploration_fraction) * priors[action] + self.exploration_fraction * (sample / total_noise)

        visit_counts = [0 for _ in range(self.network.action_size)]
        q_values = [0.0 for _ in range(self.network.action_size)]
        expanded_values: dict[int, float] = {}
        if not legal_actions:
            return SearchResult(priors, visit_counts, root_value)

        for _ in range(max(1, self.simulations)):
            total_visits = sum(visit_counts[action] for action in legal_actions)
            best_action = legal_actions[0]
            best_score = -float("inf")
            for action in legal_actions:
                score = q_values[action] + 1.25 * priors[action] * math.sqrt(total_visits + 1.0) / (1.0 + visit_counts[action])
                if score > best_score:
                    best_score = score
                    best_action = action
            if best_action not in expanded_values:
                next_hidden, reward, _ = self.network.transition(hidden, best_action, cache=False)
                _, predicted_value = self.network.predict(next_hidden)
                expanded_values[best_action] = reward + self.discount * predicted_value
            estimate = expanded_values[best_action]
            visit_counts[best_action] += 1
            q_values[best_action] += (estimate - q_values[best_action]) / visit_counts[best_action]

        total_visits = sum(visit_counts[action] for action in legal_actions)
        probabilities = [0.0 for _ in range(self.network.action_size)]
        if total_visits == 0:
            for action in legal_actions:
                probabilities[action] = 1.0 / len(legal_actions)
        else:
            for action in legal_actions:
                probabilities[action] = visit_counts[action] / total_visits
        return SearchResult(probabilities, visit_counts, root_value)

    def select_action(self, search_result: SearchResult, legal_actions: list[int], temperature: float) -> int:
        if not legal_actions:
            raise ValueError("No legal actions")
        if temperature <= 1e-6:
            return max(legal_actions, key=lambda action: search_result.action_probabilities[action])
        weights = [max(search_result.action_probabilities[action], 1e-8) ** (1.0 / temperature) for action in legal_actions]
        total = sum(weights)
        cutoff = self.rng.random()
        cumulative = 0.0
        for action, weight in zip(legal_actions, weights):
            cumulative += weight / total
            if cutoff <= cumulative:
                return action
        return legal_actions[-1]

    def remember(self, transition: Transition) -> None:
        self.replay.append(transition)

    def learn(self, updates: int) -> TrainMetrics | None:
        if len(self.replay) < self.warmup_samples or updates <= 0:
            return None
        metrics = TrainMetrics()
        for _ in range(updates):
            sample = self.replay.sample(self.rng)
            current = self._train_transition(sample)
            metrics.total_loss += current.total_loss
            metrics.value_loss += current.value_loss
            metrics.reward_loss += current.reward_loss
            metrics.policy_loss += current.policy_loss
            metrics.consistency_loss += current.consistency_loss
            self.training_steps += 1
        scale = 1.0 / updates
        metrics.total_loss *= scale
        metrics.value_loss *= scale
        metrics.reward_loss *= scale
        metrics.policy_loss *= scale
        metrics.consistency_loss *= scale
        return metrics

    def _train_transition(self, transition: Transition) -> TrainMetrics:
        hidden, rep_cache = self.network.represent(transition.observation, cache=True)
        logits, predicted_value = self.network.predict(hidden)
        next_hidden_pred, predicted_reward, transition_cache = self.network.transition(hidden, transition.action_index, cache=True)
        target_next_hidden, _ = self.network.represent(transition.next_observation, cache=False)
        _, bootstrap_value = self.network.predict(target_next_hidden)
        target_value = transition.reward if transition.done else transition.reward + self.discount * bootstrap_value

        probabilities = _softmax(logits)
        grad_logits = [probabilities[index] - transition.policy_target[index] for index in range(len(logits))]
        grad_value = 2.0 * (predicted_value - target_value)
        grad_reward = 2.0 * (predicted_reward - transition.reward)
        grad_next_hidden = [
            2.0 * (predicted - target) / len(next_hidden_pred)
            for predicted, target in zip(next_hidden_pred, target_next_hidden)
        ]

        grad_hidden_a = self.network.backward_prediction(hidden, grad_logits, grad_value, self.learning_rate)
        grad_hidden_b = self.network.backward_transition(transition_cache, grad_next_hidden, grad_reward, self.learning_rate)
        self.network.backward_representation(rep_cache, [left + right for left, right in zip(grad_hidden_a, grad_hidden_b)], self.learning_rate)

        policy_loss = -sum(target * math.log(max(probability, 1e-8)) for target, probability in zip(transition.policy_target, probabilities) if target > 0)
        value_loss = (predicted_value - target_value) ** 2
        reward_loss = (predicted_reward - transition.reward) ** 2
        consistency_loss = sum((predicted - target) ** 2 for predicted, target in zip(next_hidden_pred, target_next_hidden)) / len(next_hidden_pred)
        return TrainMetrics(policy_loss + value_loss + reward_loss + consistency_loss, value_loss, reward_loss, policy_loss, consistency_loss)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "simulations": self.simulations,
            "warmup_samples": self.warmup_samples,
            "training_steps": self.training_steps,
            "network": self.network.state_dict(),
        }
        destination.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.discount = float(payload.get("discount", self.discount))
        self.simulations = int(payload.get("simulations", self.simulations))
        self.warmup_samples = int(payload.get("warmup_samples", self.warmup_samples))
        self.training_steps = int(payload.get("training_steps", self.training_steps))
        network_state = payload.get("network")
        if isinstance(network_state, dict):
            self.network.load_state_dict(network_state)
