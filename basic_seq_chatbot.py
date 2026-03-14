from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatOpenAI()

class ChatbotState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]


def chatnode(state: ChatbotState) -> ChatbotState:

    messages = state['messages']

    response = model.invoke(messages)

    return {'messages': [response]}


graph = StateGraph(ChatbotState)


chatode = graph.add_node('chatnode', chatnode)

chatedge = graph.add_edge(START, 'chatnode')
end_edge = graph.add_edge('chatnode', END)


chatbot = graph.compile()


initial_state= {
    'messages' : [
        HumanMessage(content='What is the winter capital of Karnataka?')
    ]
}


result = chatbot.invoke(initial_state)

print(result)