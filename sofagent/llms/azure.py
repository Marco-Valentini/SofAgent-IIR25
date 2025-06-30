from loguru import logger
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_community.llms import AzureOpenAI
from langchain.schema import HumanMessage

from sofagent.llms.basellm import BaseLLM
import os


class AnyAzureLLM(BaseLLM):
    def __init__(self, model_name: str = 'gpt-4o-mini', json_mode: bool = False, *args, **kwargs):
        """
        Initialize the Azure OpenAI LLM.

        Args:
            model_name (str, optional): The name of the Azure deployment (usually matching your model).
                                          Defaults to 'gpt-3.5-turbo-1106'.
            json_mode (bool, optional): Whether to use the JSON mode of the Azure OpenAI API.
                                        Defaults to False.
            *args, **kwargs: Additional keyword arguments, which should include Azure-specific
                             parameters like openai_api_base, openai_api_version, and openai_api_key.
        """
        self.model_name = model_name
        self.json_mode = json_mode

        # Define the list of Azure models that support JSON mode.
        supported_json_models = [
            'gpt-4o',
            'gpt-4o-mini'
        ]
        if json_mode and self.model_name not in supported_json_models:
            raise ValueError(
                f"json_mode is only available for the following models: {', '.join(supported_json_models)}"
            )


        self.max_tokens: int = kwargs.get('max_tokens', 256)
        self.max_context_length: int = (
            16384 if '16k' in model_name else
            32768 if '32k' in model_name else
            128000 if model_name in ['gpt-4o-mini', 'gpt-4o'] else
            4096
        )

        # Decide which Azure model class to instantiate based on the model name.
        # Here, we assume that if the model name starts with 'text' or is a special instruct model,
        # we want to use the completions endpoint.
        if model_name.split('-')[0] == 'text' or model_name == 'gpt-3.5-turbo-instruct':
            self.model = AzureOpenAI(deployment_name=model_name, *args, **kwargs)
            self.model_type = 'completion'
        else:
            # If JSON mode is enabled, set the response format accordingly.
            if json_mode:
                logger.info("Using JSON mode of Azure OpenAI API.")
                if 'model_kwargs' in kwargs:
                    kwargs['model_kwargs']['response_format'] = {"type": "json_object"}
                else:
                    kwargs['model_kwargs'] = {"response_format": {"type": "json_object"}}
            self.model = AzureChatOpenAI(deployment_name=model_name, azure_endpoint=os.environ["OPENAI_API_BASE"], *args, **kwargs)
            self.model_type = 'chat'

    def __call__(self, prompt: str, *args, **kwargs) -> str:
        """
        Forward pass of the Azure OpenAI LLM.

        Args:
            prompt (str): The prompt to feed into the LLM.
        Returns:
            str: The Azure OpenAI LLM output.
        """
        if self.model_type == 'completion':
            # For completions, we assume that the AzureOpenAI instance provides an `invoke` method.
            # (If not, you may need to call the instance directly, e.g. self.model(prompt, ...)).
            return self.model.invoke(prompt).content.replace('\n', ' ').strip()
        else:
            # For chat, we send a list of messages.
            return self.model.invoke(
                [HumanMessage(content=prompt)]
            ).content.replace('\n', ' ').strip()
