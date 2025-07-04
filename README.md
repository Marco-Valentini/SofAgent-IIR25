# SofAgent - IIR25 - Intelligent Virtual Assistant for a Furniture Company
Repository containing the code of SofAgent, the Multi-Agent System realized in collaboration with Natuzzi Italia Innovation Center that leverages a multi-agent approach to enhance the Natuzzi customers' experience. Submitted to IIR 2025

SofAgent is an innovative software solution, developed as a Proof of Concept, that employs Artificial Intelligence and advanced Large Language Models (LLMs) to revolutionize the shopping experience for customers of a furniture company. This virtual assistant is designed to understand customer needs and intuitively and efficiently guide them through the company's extensive catalog of sofas and furnishing accessories.

<img width="1216" alt="image" src="https://github.com/user-attachments/assets/7b21ad95-d715-46b1-b442-2305a363c467" />



## Environment Installation and Setup

To run SofAgent, follow the steps below:

1.  **Prerequisites**:
    *   Python 3.10 or higher.
    *   `pip` and `venv` (or `conda`) installed.

2.  **Obtain the Source Code**:
    *   Clone the provided repository or extract the code archive into a local directory.
    ```bash
    # Example if using git:
    # git clone <URL_OF_PROVIDED_REPOSITORY>
    cd SofAgent 
    ```

3.  **Create a Python Virtual Environment (Strongly Recommended)**:
    This isolates project dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows (Command Prompt)
    # venv\Scripts\Activate.ps1 # On Windows (PowerShell)
    ```

4.  **Install Dependencies**:
    With the virtual environment active, install the necessary packages:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure API Keys**:
    *   In the `config/` folder, rename `api-config-example.json` to `api-config.json`.
    *   Open `config/api-config.json` and enter the provided API credentials (e.g., for Azure OpenAI Service):
        ```json
        {
            "api_type": "azure_openai",
            "api_base": "YOUR_AZURE_OPENAI_ENDPOINT",
            "api_key": "YOUR_AZURE_OPENAI_API_KEY",
            "api_version": "2023-12-01-preview" // Or the relevant API version
        }
        ```

6.  **Load Catalog Data**:
    *   It is essential to populate the `data/` directory with the CSV files containing the product catalog. In particular, the `processed_data_Gemini2_5Pro/` and `extracted_data_from_matrices/` subfolders must contain files like `NI_catalog_gemini2.5pro_extended.csv`, `PrezziDivani.csv`, `Abbinamenti.csv`, and the CSVs for each category of furnishing accessories.
    *   The exact paths are referenced in the configuration files within `config/tools/info_database/`.

7.  **Prepare Logos for the Web Interface**:
    *   Create a folder named `images` in the main directory of the SofAgent project.
    *   Copy the company logo files (e.g., `logo1.png`, `logo2.png`, `logo3.png` or as specified in `sofagent/pages/sofagent_ui.py`) into this `images/` folder.

## Running SofAgent

You can interact with SofAgent via two modes:

**1. Streamlit Web Interface (Recommended for Demo):**

This mode offers a graphical and interactive user experience.

*   Ensure the virtual environment is active.
*   From the main directory of the SofAgent project, run:
    ```bash
    streamlit run sofagent/pages/sofagent_ui.py
    ```
*   This command will start a local web server. Open the provided URL (usually `http://localhost:8501`) in your browser.

**2. Command-Line Interface (CLI):**

Useful for quick tests or scripted integrations.

*   Ensure the virtual environment is active.
*   From the main directory of the SofAgent project, run:
    ```bash
    python main.py --main CollaborativeChat --system_config config/systems/collaboration/chat.json --api_config config/api-config.json
    ```
*   Additional options:
    *   `--verbose DEBUG`: For more detailed log output, useful for technical tracing.
*   Interaction will occur and be displayed directly in the terminal.

## Additional Technical Notes

*   **Log Files**: Detailed interactions and debug logs are saved in the `logs/` folder (created automatically).
*   **Prompt Customization**: Agent behaviors are guided by configurable prompt templates located in `config/prompts/`.
*   **LLM Models**: The choice of LLM model (e.g., GPT-4o, GPT-4o-mini) is configurable in the JSON files within `config/agents/`.

For any questions or technical support, please contact the [development team](mailto:m.valentini7@phd.poliba.it) or consult the internal documentation.
