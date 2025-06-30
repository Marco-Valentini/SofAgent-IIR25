import tiktoken
from loguru import logger
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

from sofagent.agents.base import Agent
from sofagent.llms import AnyAzureLLM
from sofagent.utils import format_step, run_once


class Manager(Agent):
    """
    The manager agent. The manager agent is a two-stage agent, which first prompts
    the thought LLM and then prompts the action LLM.
    Its role is to understand user requests in Italian, delegate tasks to specialized
    agents by formulating natural language requests in English, and then provide a
    final response to the user in Italian.
    """

    def __init__(self, thought_config_path: str, action_config_path: str, *args, **kwargs) -> None:
        """
        Initialize the manager agent.
        It uses two LLMs: one for generating a "thought" (internal reasoning) and
        one for generating an "action" (delegation or final response).

        Args:
            thought_config_path (str): Path to the config file for the thought LLM.
            action_config_path (str): Path to the config file for the action LLM.
        """
        super().__init__(*args, **kwargs)
        self.thought_llm = self.get_LLM(thought_config_path) # LLM for generating thoughts
        self.action_llm = self.get_LLM(action_config_path)   # LLM for generating actions/final responses
        self.json_mode = self.action_llm.json_mode # Determines if the action LLM expects/outputs JSON

        # Initialize tokenizers based on whether the LLM is Azure-based or a HuggingFace model
        if isinstance(self.thought_llm, AnyAzureLLM):
            self.thought_enc = tiktoken.encoding_for_model(self.thought_llm.model_name)
        else:
            self.thought_enc = AutoTokenizer.from_pretrained(self.thought_llm.model_name)
        if isinstance(self.action_llm, AnyAzureLLM): # Corrected from thought_llm to action_llm
            self.action_enc = tiktoken.encoding_for_model(self.action_llm.model_name)
        else:
            self.action_enc = AutoTokenizer.from_pretrained(self.action_llm.model_name)

    def over_limit(self, **kwargs) -> bool:
        """
        Checks if the constructed prompt for either thought or action LLM
        would exceed their respective token limits.
        Args:
            **kwargs: Arguments to be passed to _build_manager_prompt.
        Returns:
            bool: True if either LLM's token limit would be exceeded, False otherwise.
        """
        prompt = self._build_manager_prompt(**kwargs)
        # Check token count against limits of both thought and action LLMs
        return len(self.action_enc.encode(prompt)) > self.action_llm.tokens_limit or \
               len(self.thought_enc.encode(prompt)) > self.thought_llm.tokens_limit

    @property
    def manager_prompt(self) -> PromptTemplate:
        """Returns the main prompt template for the Manager, choosing JSON version if applicable."""
        if self.json_mode:
            return self.prompts['manager_prompt_json']
        else:
            return self.prompts['manager_prompt']

    @property
    def valid_action_example(self) -> str:
        """Returns examples of valid actions, formatted for JSON or raw string mode."""
        if self.json_mode:
            return self.prompts['valid_action_example_json'].replace('{finish}', self.prompts['finish_json'])
        else:
            return self.prompts['valid_action_example'].replace('{finish}', self.prompts['finish'])

    @property
    def fewshot_examples(self) -> str:
        """Returns few-shot examples to guide the LLM, if provided in prompts."""
        if 'fewshot_examples' in self.prompts:
            return self.prompts['fewshot_examples']
        else:
            return ''

    @property
    def hint(self) -> str:
        """Returns a hint message, typically used when the agent is about to Finish."""
        if 'hint' in self.prompts:
            return self.prompts['hint']
        else:
            return ''

    @run_once # Decorator to ensure this logging happens only once per run if needed
    def _log_prompt(self, prompt: str) -> None:
        """Logs the full manager prompt for debugging purposes."""
        logger.debug(f'Manager Prompt: {prompt}')

    def _build_manager_prompt(self, **kwargs) -> str:
        """
        Constructs the full prompt string for the Manager's LLMs.
        Args:
            **kwargs: Key-value pairs to format into the prompt template
                      (e.g., history, scratchpad, max_step).
        Returns:
            str: The fully formatted prompt.
        """
        return self.manager_prompt.format(
            examples=self.fewshot_examples, # Injects few-shot examples into the prompt
            **kwargs
        )

    def _prompt_thought(self, **kwargs) -> str:
        """
        Generates a 'thought' by prompting the thought LLM.
        Args:
            **kwargs: Arguments for building the manager prompt.
        Returns:
            str: The formatted thought string from the LLM.
        """
        thought_prompt = self._build_manager_prompt(**kwargs)
        self._log_prompt(thought_prompt) # Log the prompt for debugging (runs once per session due to decorator)
        thought_response = self.thought_llm(thought_prompt)
        return format_step(thought_response) # Clean up the response string

    def _prompt_action(self, **kwargs) -> str:
        """
        Generates an 'action' (or final response) by prompting the action LLM.
        Args:
            **kwargs: Arguments for building the manager prompt.
        Returns:
            str: The formatted action/response string from the LLM.
        """
        action_prompt = self._build_manager_prompt(**kwargs)
        # self._log_prompt(action_prompt) # Typically, prompt structure is similar, might not need to log again
        action_response = self.action_llm(action_prompt)
        return format_step(action_response) # Clean up the response string

    def forward(self, stage: str, *args, **kwargs) -> str:
        """
        Main entry point for the Manager agent, called by the system.
        Directs to either thought generation or action generation based on the 'stage'.
        Args:
            stage (str): The current stage of operation ('thought' or 'action').
            **kwargs: Additional arguments passed to the prompt building functions.
        Returns:
            str: The generated thought or action string.
        Raises:
            ValueError: If an unsupported stage is provided.
        """
        # based on the stage, the manager can think or act
        if stage == 'thought':
            return self._prompt_thought(**kwargs)
        elif stage == 'action':
            return self._prompt_action(**kwargs)
        else:
            raise ValueError(f"Unsupported stage: {stage}")