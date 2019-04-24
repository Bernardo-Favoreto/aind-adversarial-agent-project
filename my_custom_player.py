from sample_players import DataPlayer
import random, math, copy

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
       # print(self.data)
       # print(state.ply_count)
        def score(self, state):
            own_loc = state.locs[self.player_id]
            opp_loc = state.locs[1 - self.player_id]
            own_liberties = state.liberties(own_loc)
            opp_liberties = state.liberties(opp_loc)
            return len(own_liberties) - len(opp_liberties)
        def alpha_beta_search (state, player_id, depth):

            # Max Value function
            def max_value(state, alpha, beta, depth):
                if state.terminal_test():
                    return state.utility(player_id)
                if depth <= 0: return score(self, state)
                v = float("-inf") 
                for action in state.actions():
                    v = max(v, min_value(state.result(action), alpha, beta, depth - 1))
                    if v >= beta:
                        return v
                    alpha = max(alpha, v)
                return v

            #Min Value function
            def min_value(state, alpha, beta, depth):
                if state.terminal_test():
                    return state.utility(player_id)
                if depth <= 0: return score(self, state)
                v = float("inf")
                for action in state.actions():
                    v = min(v, max_value(state.result(action), alpha, beta, depth - 1))
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
                return v
            # "Decision" function. Return best possible move.
            alpha = float("-inf")
            beta = float("inf")
            v = max_value(state, alpha, beta, depth)
            best_score = float("-inf")
            best_move = None
            for action in state.actions():
                v = min_value(state.result(action), alpha, beta, depth - 1)
                if v > best_score:
                    best_score = v
                    best_move = action
            return best_move

        if state.ply_count < 4:
            if state in self.data:
                self.queue.put(self.data[state])
            else:
                self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 5
            for depth in range (1, depth_limit + 1):
                #print(state)
                best_move = alpha_beta_search(state, self.player_id, depth)
                #print("oi")
            self.queue.put(best_move)

iter_limit = 150

class MCTS_player(DataPlayer):
    def MCTS(self, state):
        root = MCTS_node(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        for _ in range(iter_limit):
            child = tree_policy(root)
            if not child:
                continue
            reward = simulation(child.state)
            backprop(child, reward)

        index = root.children.index(best_child(root))
        return root.children_actions[index] # Choose the action using the index

    def get_action(self, state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.MCTS(state))

class MCTS_node():
    def __init__(self, state, parent = None):
        self.visits = 1 # Every node is initiated with number of visits = 1
        self.reward = 0
        self.state = state
        self.children = []
        self.children_actions = []
        self.parent = parent

    def add_child(self, child_state, action):
        child = MCTS_node(child_state, self) # Self is the parent of this node
        self.children.append(child)
        self.children_actions.append(action)

    def update_tree(self, reward):
        self.visits += 1
        self.reward += reward

    def is_fully_explored(self):
        return len(self.children_actions) == len(self.state.actions()) #if the child actions are the same as the current state, then it is fully explored.

def tree_policy(node):
    while not node.state.terminal_test():
        if not node.is_fully_explored():
            return expand(node)
        node = best_child(node)
    return node

def expand(node):
    attempted_actions = node.children_actions
    legal_actions = node.state.actions()
    for action in legal_actions:
        if action not in attempted_actions:
            new_state = node.state.result(action) # The new state is the one after the action is executed
            node.add_child(new_state, action) # Add this new state as a child node for current root node
            return node.children[-1]
    return None

def best_child(node):
    best_score = float("-inf") # Set the worst possible score first
    best_children = []
    for child in node.children:
        exploit = child.reward / child.visits # Exploitation part of the equation
        explore = math.sqrt(2 * math.log(node.visits) / child.visits)
        score = exploit + explore
        if score == best_score:
            best_children.append(child)
        elif score > best_score:
            best_children = [child]
            best_score = score

    return random.choice(best_children) # After we know which are the best possible children, the decision is randomized

def simulation(state):
    initial_state = copy.deepcopy(state) # 
    while not state.terminal_test():
        action = random.choice(state.actions()) # Choose a random action from the legal actions
        state = state.result(action) # New state is the one after action is executed

    return -1 if state._has_liberties(initial_state.player()) else 1 # Return 1 for winning games, -1 otherwise

def backprop(node, reward):
    while node != None:
        node.update_tree(reward) # Update the tree based on the reward value
        node = node.parent # Going "up" the tree
        reward *= -1
