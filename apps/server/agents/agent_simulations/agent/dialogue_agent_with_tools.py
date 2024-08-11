from typing import List, Optional

from xagent import XAgent, XAgentExecutor, XAgentConfig, XAgentCallbackHandler, XAgentErrorHandling
from xagent.memory import XAgentMemory
from xagent.output_parser import XAgentOutputParser
from config import Config
from services.run_log import RunLogsManager
from typings.agent import AgentWithConfigsOutput


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        agent_with_configs: AgentWithConfigsOutput,
        system_message: SystemMessage,
        model: XAgent,  # Replace ChatOpenAI with XAgent
        tools: List[any],
        session_id: str,
        sender_name: str,
        is_memory: bool = False,
        run_logs_manager: Optional[RunLogsManager] = None,
        **tool_kwargs,
    ) -> None:
        super().__init__(name, agent_with_configs, system_message, model)
        self.tools = tools
        self.session_id = session_id
        self.sender_name = sender_name
        self.is_memory = is_memory
        self.run_logs_manager = run_logs_manager

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """

        memory: XAgentMemory

        # FIXME: This is a hack to get the memory working
        if self.is_memory:
            memory = XAgentMemory(
                session_id=self.session_id,
                url=Config.ZEP_API_URL,
                api_key=Config.ZEP_API_KEY,
                memory_key="chat_history",
                return_messages=True,
            )

            memory.human_name = self.sender_name
            memory.ai_name = self.agent_with_configs.agent.name
            memory.auto_save = False
        # else:
        #     memory = ConversationBufferMemory(
        #         memory_key="chat_history", return_messages=True
        #     )

        callbacks = []

        if self.run_logs_manager:
            self.model.callbacks = [self.run_logs_manager.get_agent_callback_handler()]
            callbacks.append(self.run_logs_manager.get_agent_callback_handler())

        agent = XAgent(
            model_name="gpt-3.5-turbo",
            temperature=0,
            tools=self.tools,
            system_message=self.system_message.content,
            output_parser=XAgentOutputParser(),
            max_iterations=5,
            verbose=True,
            handle_parsing_errors="Check your output and make sure it conforms!",
            memory=memory,
            callbacks=callbacks,
        )

        agent_executor = XAgentExecutor(agent=agent, tools=self.tools, verbose=True)

        prompt = "\n".join(self.message_history + [self.prefix])

        res = agent_executor.run(input=prompt)

        # FIXME: is memory
        # memory.save_ai_message(res)

        message = AIMessage(content=res)

        return message.content
