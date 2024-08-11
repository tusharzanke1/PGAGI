
from xagent import XAgent, XAgentClient, XAgentConfig, XAgentEvalConfig
from xagent.tools import XSerpGoogleSearch
from langsmith import Client

# Refactored function to initialize the XAgent
def xagent_factory():
    # Initialize the XAgent with necessary configurations
    xagent = XAgent(
        model_name="gpt-3.5-turbo",
        temperature=0,
        tools=[XSerpGoogleSearch()],
        system_message="Your system message here",
        output_parser="Your output parser here",
        max_iterations=5,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms!"
    )
    return xagent

# Initialize the agent
agent = xagent_factory()

# Initialize the Langsmith client (assuming it can interact with XAgent)
client = Client()

# Evaluation configuration using XAgent's configuration class
eval_config = XAgentEvalConfig(
    evaluators=[
        "qa",
        XAgentEvalConfig.Criteria("helpfulness"),
        XAgentEvalConfig.Criteria("conciseness"),
    ],
    input_key="input",
    eval_llm=XAgent(model_name="gpt-3.5-turbo", temperature=0.5),
)

# Running evaluation on the dataset with XAgent
chain_results = client.run_on_dataset(
    dataset_name="test-dataset",
    llm_or_chain_factory=xagent_factory,
    evaluation=eval_config,
    concurrency_level=1,
    verbose=True,
)
