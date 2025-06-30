from typing import Any, Union, List, Optional, Tuple
from loguru import logger

from sofagent.agents.base import ToolAgent
from sofagent.tools import InfoSofabase # Tool for accessing sofa-specific data
from sofagent.utils import read_json, get_rm, parse_action, format_step


class SofaAnalyzer(ToolAgent):
    """
    Agent specialized in analyzing and retrieving detailed information about sofas.
    It can fetch sofa IDs, general information, and specific price/configuration details.
    """
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        """
        Initializes the SofaAnalyzer agent.
        Args:
            config_path (str): Path to the agent's JSON configuration file.
        """
        super().__init__(*args, **kwargs)
        config = read_json(config_path)
        tool_config: dict[str, dict] = get_rm(config, 'tool_config', {})
        self.get_tools(tool_config) # Load and initialize tools (InfoSofabase)
        self.max_turns = get_rm(config, 'max_turns', 4) # Max internal LLM calls before forcing a finish
        self.llm = self.get_LLM(config=config) # LLM instance for this agent
        self.json_mode = self.llm.json_mode # Whether the LLM expects/outputs JSON
        self.reset()

    @staticmethod
    def required_tools() -> dict[str, type]:
        """Specifies the tools required by this agent."""
        return {'info_sofa': InfoSofabase}

    @property
    def info_retriever(self) -> InfoSofabase:
        """Provides access to the InfoSofabase tool instance."""
        return self.tools['info_sofa']

    # --- Prompts (Directly from self.prompts loaded by base class) ---
    @property
    def sofa_analyzer_prompt_template(self) -> str:
        """Returns the main prompt template for the SofaAnalyzer, choosing JSON version if applicable."""
        return self.prompts['sofa_analyzer_prompt_json'] if self.json_mode else self.prompts['sofa_analyzer_prompt']

    @property
    def hint_prompt(self) -> str:
        """Returns a hint message, typically used when the agent is about to Finish or nearing max_turns."""
        return self.prompts.get('sofa_analyzer_hint', '') # Use .get for optional hints

    def _build_prompt(self, manager_request_str: str) -> str:
        """
        Builds the full prompt string to be sent to the SofaAnalyzer's LLM.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The fully formatted prompt.
        """
        # The prompt template itself now includes placeholders for examples and few-shot directly
        return self.sofa_analyzer_prompt_template.format(
            manager_request=manager_request_str,
            history=self.history, # Formatted history of previous commands/observations in this sub-task
            hint=self.hint_prompt if len(self._history) + 1 >= self.max_turns else ''
        )

    def _prompt_sofa_analyzer(self, manager_request_str: str) -> str:
        """
        Sends the built prompt to the LLM and gets its proposed command.
        Args:
            manager_request_str (str): The natural language request from the Manager.
        Returns:
            str: The command string proposed by the LLM.
        """
        prompt = self._build_prompt(manager_request_str=manager_request_str)
        command_from_llm = self.llm(prompt)
        return command_from_llm

    def command(self, command_str: str) -> None:
        """
        Parses the LLM's command string, executes the corresponding tool action,
        and records the observation.
        Args:
            command_str (str): The command string from the LLM.
        """
        logger.debug(f'SofaAnalyzer LLM proposed command: {command_str}')
        log_head = '' # For formatting Streamlit logs
        action_type, argument = parse_action(command_str, json_mode=self.json_mode)
        observation = ""

        if action_type.lower() == 'getsofaid':
            if isinstance(argument, str) and argument.strip():
                sofa_name = argument.strip()
                try:
                    observation = self.info_retriever.get_sofa_id_by_name(sofa_name=sofa_name)
                    log_head = f':violet[Tool Call: GetSofaID for] :red[{sofa_name}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling GetSofaID for '{sofa_name}': {e}"
                    logger.error(f"GetSofaID error: {e}")
            else:
                observation = "Invalid argument for GetSofaID. Expected a sofa name string."

        elif action_type.lower() == 'sofainfo':
            items_to_query = []
            if isinstance(argument, str): # Handles "item1,item2" or "item1"
                items_to_query = [item.strip() for item in argument.split(',')]
            elif isinstance(argument, list) and argument: # Handles JSON like [["item1"], ["item2"]] or ["item1,item2"]
                if isinstance(argument[0], list): # Handles [["item1"], ["item2"]]
                    items_to_query = [str(item[0]) for item_list in argument for item in item_list if isinstance(item, str)]
                elif isinstance(argument[0], str): # Handles ["item1,item2,item3"]
                    items_to_query = [item.strip() for item in argument[0].split(',')]

            if items_to_query:
                try:
                    observation = self.info_retriever.sofa_info(sofa_ids_or_names=items_to_query)
                    log_head = f':violet[Tool Call: SofaInfo for] :red[{items_to_query}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling SofaInfo for '{items_to_query}': {e}"
                    logger.error(f"SofaInfo error: {e}")
            else:
                observation = "Invalid or empty argument for SofaInfo. Expected ID/Name or comma-separated list."
        elif action_type.lower() == 'sofapriceandconfigs':
            if isinstance(argument, str) and argument.strip():
                sofa_id = argument.strip()
                try:
                    # Tool method for fetching predefined configurations with prices and seats
                    observation = self.info_retriever.get_sofa_predefined_configs(sofa_id_str=sofa_id)
                    log_head = f':violet[Tool Call: SofaPriceAndConfigs for sofa ID] :red[{sofa_id}]:violet[...]\n- '
                except Exception as e:
                    observation = f"Error calling get_sofa_predefined_configs for '{sofa_id}': {e}"
                    logger.error(f"get_sofa_predefined_configs error: {e}")
            else:
                observation = "Invalid argument for SofaPriceAndConfigs. Expected a sofa ID string."

        elif action_type.lower() == 'finish':
            observation = self.finish(results=argument)  # Argument is the result string for the Manager
            log_head = ':violet[Finish with results for Manager]:\n- '
        else:
            observation = f'Unknown command type from SofaAnalyzer LLM: {action_type}.'

        logger.debug(f'SofaAnalyzer Observation: {observation}')
        self.observation(str(observation), log_head)  # Logs to system (and Streamlit if web_demo)
        turn = {'command': command_str, 'observation': str(observation)}
        self._history.append(turn)

    def forward(self, manager_request: str, *args: Any, **kwargs: Any) -> str:
        """
        Main execution loop for the SofaAnalyzer. It iteratively calls its LLM
        to decide on actions (tool calls) until it calls Finish or hits max_turns.
        Args:
            manager_request (str): The initial request from the Manager.
        Returns:
            str: The final result to be passed back to the Manager.
        """
        self._history = []  # Clear history for this specific sub-task from Manager

        while not self.is_finished():
            if len(self._history) >= self.max_turns: # Check if max turns are reached
                logger.warning(
                    f"{self.__class__.__name__} reached max_turns ({self.max_turns}) without a Finish command.")
                if self._history: # Return last known observation if max_turns is hit
                    return f"Max turns reached. Last observation: {self._history[-1]['observation']}"
                else:
                    return "Max turns reached, but no actions were taken or observations made."

            current_manager_request_for_prompt = manager_request

            command_str = self._prompt_sofa_analyzer(manager_request_str=current_manager_request_for_prompt)
            self.command(command_str)  # This appends to self.history and might set self.finished

            if self.is_finished():  # Exit loop if Finish command was executed
                break

        if not self.finished: # Fallback if loop exits for other reasons without finishing
            logger.error(
                f"{self.__class__.__name__} exited loop but was not marked as finished and has no history. This should not happen.")
            return "An unexpected state occurred in the agent's execution."

        return self.results if self.results is not None else "No results were finalized by the agent."

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Entry point for the system to call this agent.
        Args:
            argument (Any): The natural language request from the Manager.
            json_mode (bool): Indicates if the LLM should operate in JSON mode.
        Returns:
            str: The final result from the agent's processing.
        """
        self.reset()
        self.json_mode = json_mode
        if not isinstance(argument, str) or not argument.strip():
            return "Invalid argument for SofaAnalyzer. Expected a natural language request string from Manager."

        manager_request_str = argument.strip()
        return self.forward(manager_request=manager_request_str)

    def __call__(self, manager_request: str, *args: Any, **kwargs: Any) -> str:
        """Allows the agent to be called like a function."""
        self.reset()
        return self.forward(manager_request=manager_request, *args, **kwargs)

# Debug script
if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from sofagent.utils import init_api, read_json, read_prompts
    import sys
    import os

    # --- Setup for local script execution ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    try:
        # Define paths to configuration files
        api_config_file = os.path.join(project_root, 'config/api-config.json')
        agent_prompts_file = os.path.join(project_root, 'config/prompts/agent_prompt/sofa_analyzer.json')
        agent_config_file = os.path.join(project_root, 'config/agents/sofa_analyzer.json')

        # Manually ensure the tool config points to the correct file for predefined configurations during test
        tool_config_path = os.path.join(project_root, 'config/tools/info_database/sofa.json')
        tool_config = read_json(tool_config_path) # Load existing tool config
        # Add or override the path for predefined configurations data
        tool_config['predefined_configs_info'] = 'data/passport_data/ConfigurazioniPredefiniteDivani_ConPrezziESedute.csv'


        init_api(read_json(api_config_file)) # Initialize API (e.g., Azure OpenAI)
        prompts = read_prompts(agent_prompts_file) # Load prompt templates

        # Dummy system class for testing purposes
        class DummySystem:
            def __init__(self):
                self.web_demo = False
                self.task_type = "analyze_sofa" # Task type for prompt formatting
                self.data_sample = None

            def log(self, m, agent=None, logging=True): # Simplified logging for console
                role_name = agent.__class__.__name__ if agent else 'AgentInternal'
                if "Tool Call" in m or "Finish" in m or "Manager Request" in m:
                    logger.info(f"({role_name}): {m}")
                else:
                    logger.debug(f"({role_name}): {m}")

        dummy_sys = DummySystem()
        # Partial prompt formatting if task_type is an input variable
        for pname, ptemplate in prompts.items():
            if isinstance(ptemplate, PromptTemplate) and 'task_type' in ptemplate.input_variables:
                prompts[pname] = ptemplate.partial(task_type=dummy_sys.task_type)

        # Instantiate the SofaAnalyzer agent
        sofa_analyzer = SofaAnalyzer(config_path=agent_config_file, prompts=prompts, system=dummy_sys)

        # Example test requests from the Manager
        test_requests_from_manager = [
            "What is the ID for sofa 'Adam'?",
            "Tell me about sofa ID 3242.",
            "What are the configurations, prices, and number of seats for sofa Iago, ID 2954?",
            "Show me price options and seats for the Philo sofa.",
            "Provide general features for the Herman sofa, and also get the price and seat configurations for sofa ID 2957.",
            "What are the price options for sofa ID 99999?"
        ]

        # Loop through test requests and invoke the agent
        for i, manager_req in enumerate(test_requests_from_manager):
            logger.info(f"--- SofaAnalyzer Test Case {i + 1} ---")
            logger.info(f"Manager Request to SofaAnalyzer: \"{manager_req}\"")
            response_to_manager = sofa_analyzer.invoke(argument=manager_req, json_mode=sofa_analyzer.json_mode)
            logger.success(f"SofaAnalyzer's Final Response (to Manager):\n{response_to_manager}\n")
            # Agent's reset is called within invoke()

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}. CWD: {os.getcwd()}")
        logger.error(f"Please ensure all paths are correct. Project root was assumed to be: {project_root}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)