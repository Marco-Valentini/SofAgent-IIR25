import json
from typing import Any, Optional
from loguru import logger

from sofagent.systems.base import System
from sofagent.agents import Agent, Manager, SofaAnalyzer, FurnAnalyzer, MatchExpert, SofaSearcher, FurnSearcher
from sofagent.utils import parse_answer, parse_action, format_chat_history


class CollaborationSystem(System):
    """
    Orchestrates a collaborative multi-agent system where a Manager agent delegates
    tasks to specialized agents (Analyzers, Searchers, MatchExpert) to fulfill user requests.
    Primarily designed for a 'chat' task.
    """

    @staticmethod
    def supported_tasks() -> list[str]:
        """Returns a list of tasks supported by this system."""
        return ['chat']

    def init(self, *args, **kwargs) -> None:
        """
        Initializes the CollaborationSystem based on its configuration.
        Sets up the maximum steps for interaction and initializes all configured agents.
        """
        self.max_step: int = self.config.get('max_step', 10) # Maximum interaction steps for the Manager
        assert 'agents' in self.config, 'Agents configuration is required in the system config.'
        self.init_agents(self.config['agents'])
        self.manager_kwargs = { # Keyword arguments to pass to the Manager during its operation
            'max_step': self.max_step,
        }

    def init_agents(self, agents: dict[str, dict]) -> None:
        """
        Initializes all agents specified in the system configuration.
        Args:
            agents (dict[str, dict]): A dictionary where keys are agent names (matching class names)
                                      and values are their respective configuration dictionaries.
        """
        self.agents: dict[str, Agent] = dict()
        for agent_name, agent_config in agents.items():
            try:
                agent_class = globals()[agent_name] # Dynamically get agent class by name
                assert issubclass(agent_class, Agent), f'Agent {agent_name} is not a subclass of Agent.'
                # Instantiate agent with its config and common agent arguments
                self.agents[agent_name] = agent_class(**agent_config, **self.agent_kwargs)
            except KeyError:
                raise ValueError(f'Agent {agent_name} is not supported or not found in globals.')
        assert 'Manager' in self.agents, 'A Manager agent is required for the CollaborationSystem.'

    @property
    def manager(self) -> Optional[Manager]:
        """Provides access to the Manager agent instance."""
        if 'Manager' not in self.agents:
            return None
        return self.agents['Manager']

    # Properties to access specialized agents
    @property
    def sofa_analyzer(self) -> Optional[SofaAnalyzer]:
        """Provides access to the SofaAnalyzer agent instance, if configured."""
        return self.agents.get('SofaAnalyzer')

    @property
    def furn_analyzer(self) -> Optional[FurnAnalyzer]:
        """Provides access to the FurnAnalyzer agent instance, if configured."""
        return self.agents.get('FurnAnalyzer')

    @property
    def sofa_searcher(self) -> Optional[SofaSearcher]:
        """Provides access to the SofaSearcher agent instance, if configured."""
        return self.agents.get('SofaSearcher')

    @property
    def furn_searcher(self) -> Optional[FurnSearcher]:
        """Provides access to the FurnSearcher agent instance, if configured."""
        return self.agents.get('FurnSearcher')

    @property
    def match_expert(self) -> Optional[MatchExpert]:
        """Provides access to the MatchExpert agent instance, if configured."""
        return self.agents.get('MatchExpert')

    def reset(self, clear_session_history: bool = False, *args, **kwargs) -> None:
        """
        Resets the system state for a new interaction, including step count and chat history.
        Also resets all configured agents.
        Args:
            clear_session_history (bool): If True, clears the internal chat history.
        """
        super().reset(clear_session_history=clear_session_history, *args, **kwargs)
        self.step_n: int = 1 # Reset step counter for Manager's interactions
        if clear_session_history:
            if self.task == 'chat':
                if not hasattr(self, '_chat_history') or self._chat_history is None:
                    self._chat_history = []
                else:
                    self._chat_history.clear()

        # Reset each individual agent
        for agent_instance in self.agents.values():
            if hasattr(agent_instance, 'reset') and callable(agent_instance.reset):
                agent_instance.reset()

    def add_chat_history(self, chat: str, role: str) -> None:
        """
        Adds a message to the internal chat history.
        Args:
            chat (str): The message content.
            role (str): The role of the speaker (e.g., 'user', 'system').
        """
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        if not hasattr(self, '_chat_history'):
            self._chat_history = []
        self._chat_history.append((chat, role))

    @property
    def chat_history(self) -> str:
        """Formats and returns the current chat history as a string for the Manager's prompt."""
        assert self.task == 'chat', 'Chat history is only available for chat task.'
        if not hasattr(self, '_chat_history'):
            self._chat_history = []
        return format_chat_history(self._chat_history)

    def is_halted(self) -> bool:
        """
        Checks if the system should halt due to exceeding max steps or token limits for the Manager.
        Returns:
            bool: True if the system should halt, False otherwise.
        """
        # Halt if max_step reached OR manager's prompt (with current scratchpad) exceeds token limits,
        # AND the system hasn't already finished.
        return ((self.step_n > self.max_step) or \
                (self.manager and self.manager.over_limit(scratchpad=self.scratchpad, **self.manager_kwargs))) \
               and not self.finished

    def _parse_answer(self, answer: Any = None) -> dict[str, Any]:
        """
        Parses the final answer from the Manager before finishing.
        Args:
            answer (Any, optional): The answer to parse. Defaults to self.answer.
        Returns:
            dict[str, Any]: A dictionary containing the parsed answer and validity status.
        """
        if answer is None:
            answer = self.answer
        return parse_answer(type=self.task, answer=answer, gt_answer=self.gt_answer if self.task != 'chat' else '',
                            json_mode=self.manager.json_mode if self.manager else False, **self.kwargs)

    def think(self):
        """Executes the Manager's 'thought' stage to plan the next action."""
        logger.debug(f'Step {self.step_n}:')
        self.scratchpad += f'\nThought {self.step_n}:' # Append thought marker to scratchpad
        # Get thought from Manager LLM
        thought = self.manager(scratchpad=self.scratchpad, stage='thought', **self.manager_kwargs)
        self.scratchpad += ' ' + thought # Append thought content to scratchpad
        self.log(f'**Thought {self.step_n}**: {thought}', agent=self.manager) # Log the thought

    def act(self) -> tuple[str, Any]:
        """
        Executes the Manager's 'action' stage to decide on the next delegation or to finish.
        Returns:
            tuple[str, Any]: The action type and its argument.
        """
        # If it's the last allowed step, add a hint to the scratchpad for the Manager.
        if self.max_step == self.step_n:
            self.scratchpad += f'\nHint: {self.manager.hint if self.manager else ""}'
        self.scratchpad += f'\nValid action example: {self.manager.valid_action_example if self.manager else ""}:'
        self.scratchpad += f'\nAction {self.step_n}:' # Append action marker
        # Get action from Manager LLM
        action = self.manager(scratchpad=self.scratchpad, stage='action', **self.manager_kwargs)
        self.scratchpad += ' ' + action # Append action content to scratchpad
        action_type, argument = parse_action(action, json_mode=self.manager.json_mode if self.manager else False)
        logger.debug(f'Action {self.step_n}: {action}')
        return action_type, argument

    def execute(self, action_type: str, argument: Any):
        """
        Executes the action decided by the Manager.
        If the action is 'Finish', it finalizes the system's answer.
        If it's a delegation action, it calls the appropriate specialist agent.
        Args:
            action_type (str): The type of action to execute (e.g., 'DelegateToSofaAnalyzer', 'Finish').
            argument (Any): The argument for the action (e.g., NL request for an agent, or final response).
        """
        log_head = '' # For Streamlit logging prefix
        observation = '' # Result of the action
        action_type_lower = action_type.lower()

        # Mapping from Manager's delegation action types to agent instances
        agent_map = {
            'delegatetosofaanalyzer': self.sofa_analyzer,
            'delegatetosofasearcher': self.sofa_searcher,
            'delegatetofurnanalyzer': self.furn_analyzer,
            'delegatetofurnsearcher': self.furn_searcher,
            'delegatetomatchexpert': self.match_expert,
        }

        if action_type_lower == 'finish':
            parse_result = self._parse_answer(argument) # Validate and parse the final answer
            if parse_result['valid']:
                observation = self.finish(parse_result['answer']) # Mark system as finished
                log_head = ':violet[Finish with answer]:\n- '
            else: # Handle invalid final answer format
                valid_example_str = self.manager.valid_action_example if self.manager else "See prompt examples."
                observation = f'{parse_result.get("message", "Parsing error.")} Valid Action examples are {valid_example_str}.'

        elif action_type_lower in agent_map: # If it's a delegation action
            agent_to_call = agent_map[action_type_lower]
            agent_name = agent_to_call.__class__.__name__ if agent_to_call else "UnknownAgent"

            if agent_to_call is None: # Check if the specialist agent is configured
                observation = f'{agent_name} is not configured in the system.'
            elif not isinstance(argument, str) or not argument.strip(): # Argument for delegation must be a string request
                observation = f"Invalid argument for {action_type}. Expected a non-empty string request."
            else:
                # Log the delegation call to the specialist agent
                self.log(f':violet[Calling] :red[{agent_name}] :violet[with request:] :blue[[{argument}]]:violet[...]',
                         agent=self.manager, logging=False) # `logging=False` avoids double logging if agent also logs

                # Invoke the specialist agent with the Manager's request
                observation = agent_to_call.invoke(argument=argument, json_mode=self.manager.json_mode if self.manager else False)

                log_head = f':violet[Response from] :red[{agent_name}]:violet[:]\n- '
        else:
            # Handle any unknown or invalid action types
            valid_example_str = self.manager.valid_action_example if self.manager else "See prompt examples."
            observation = f'Invalid Action type: {action_type}. Valid Actions start with "DelegateTo..." or "Finish". Examples: {valid_example_str}.'

        self.scratchpad += f'\nObservation: {observation}' # Add agent's observation to scratchpad for Manager
        logger.debug(f'Observation: {observation}')
        self.log(f'{log_head}{observation}', agent=self.manager, logging=False) # Log observation for system/UI

    def step(self):
        """Performs one full think-act-execute cycle of the Manager."""
        self.think()
        action_type, argument = self.act()
        self.execute(action_type, argument)
        self.step_n += 1 # Increment step counter

    def forward(self, user_input: Optional[str] = None, reset_session_history: bool = False) -> Any:
        """
        Main entry point for the system to process a user request.
        Args:
            user_input (Optional[str]): The user's input for a chat task.
            reset_session_history (bool): Whether to clear previous chat history.
        Returns:
            Any: The final answer from the system.
        """
        self.reset(clear_session_history=reset_session_history) # Reset system state
        self.manager_kwargs['history'] = self.chat_history # Pass current chat history to Manager
        if self.task == 'chat':
            assert user_input is not None, 'User input is required for chat task.'
            self.add_chat_history(user_input, role='user') # Add user's new message
            self.manager_kwargs['history'] = self.chat_history # Update history for Manager
        else: # For other potential tasks (not 'chat')
            self.manager_kwargs['input'] = self.input

        if self.web_demo: # If running in web demo mode, clear previous web logs
            self.clear_web_log()

        # Main interaction loop: continues until the system is finished or halted
        while not self.is_finished() and not self.is_halted():
            self.step()

        # If it's a chat task and an answer was generated, add it to the history
        if self.task == 'chat' and self.answer is not None:
            self.add_chat_history(self.answer, role='system') # 'system' role for SofAgent's final response

        return self.answer

    def get_response_for_prompt(self, prompt: str) -> str:
        """
        A dedicated method for getting a single response for a given prompt,
        ensuring the session history is reset for independent evaluation.
        Args:
            prompt (str): The user prompt/question.
        Returns:
            str: The final answer from the manager.
        """
        # Call forward with reset_session_history=True to treat each prompt independently
        return self.forward(user_input=prompt, reset_session_history=True)

    def chat(self) -> None:
        """Provides a simple command-line interface for chatting with the system."""
        assert self.task == 'chat', 'Chat task is required for chat method.'
        print("Start chatting with the system. Type 'exit' or 'quit' to end the conversation.")
        self.reset(clear_session_history=True) # Start a fresh chat session
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            response = self(user_input=user_input, reset_session_history=False) # Process user input
            print(f"System: {response}")