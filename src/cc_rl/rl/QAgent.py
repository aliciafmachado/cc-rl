import torch
from cc_rl.rl.Agent import Agent
from cc_rl.rl.QModel import QModel
from torch.utils.data.dataloader import default_collate

class QAgent(Agent):
    """
    Reinforcement learning agent that uses only Q-learning to find the best tree path.
    """
    def __init__(self, environment):
        super().__init__(environment)
        self.model = QModel(environment.classifier_chain.n_labels + 1, self.device)

    def train(self, nb_sim, nb_paths, epochs, batch_size=64, learning_rate=1e-3, verbose=True):
        # We do multiple simulations
        for sim in range(nb_sim):
            self.experience_environment(nb_paths, batch_size)
            self.train_once(epochs, learning_rate, verbose)

    def predict(self):
        pass

    def experience_environment(self, nb_paths, batch_size):
        """
        In this method the model is used to predict the best path for a total of
        nb_paths paths. For each decision the model takes, the state is recorded.
        The resul is then store in the variable self.data_loader
        """
        
        # Resetting history
        actions_history = []
        probas_history = []
        next_probas = []
        next_actions = []
        final_values = []

        # Each path should go until the end
        depth = self.environment.classifier_chain.n_labels

        for i in range(nb_paths):
            # Get current state from environment
            next_proba, action_history, proba_history = self.environment.reset()

            for j in range(depth):
                # Getting the path information as torch tensors
                action_history = torch.tensor(action_history).float()
                proba_history = torch.tensor(proba_history).float()
                next_proba = torch.tensor(next_proba[0]).float()

                # Choosing the next action using the agent
                next_action = self.model.choose_action(action_history, proba_history, next_proba)
                next_proba, action_history, proba_history, final_value, end = self.environment.step(next_action)

                # Adding past actions to the history
                actions_history += [action_history]
                probas_history += [proba_history]
                next_probas += [next_proba[0]]
                next_actions += [next_action]

            # Updating the history for the final values
            for j in range(depth):
                final_values += [final_value]

        # Updating data loader to train the network
        actions_history = torch.tensor(actions_history).float()
        probas_history = torch.tensor(probas_history).float()
        next_actions = torch.tensor(next_actions).float()
        next_probas = torch.tensor(next_probas).float()
        final_values = torch.tensor(final_values).float()

        dataset = torch.utils.data.TensorDataset(actions_history, probas_history,
                                                 next_actions, next_probas, final_values)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                                                    #    collate_fn=lambda x: default_collate(x).to(self.device))

    def train_once(self, epochs, learning_rate, verbose):  
        # Start training
        self.model.train()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for i, data in enumerate(self.data_loader):
                actions_history, probas_history, next_probas, next_actions, final_values = [d.to(self.device) for d in data]

                # Calculate Q value for each test case
                predict = self.model(actions_history, probas_history, next_probas, next_actions).reshape(-1)

                # Apply loss function
                loss = loss_fn(predict, final_values)
                
                # Brackprop and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose:
                    print ('Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch+1, epochs, i+1, len(self.data_loader), 
                        loss.item() / self.data_loader.batch_size))
