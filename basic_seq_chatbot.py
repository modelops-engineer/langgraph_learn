from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

model = ChatOpenAI()

class ChatbotState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]


def chatnode(state: ChatbotState) -> ChatbotState:

    messages = state['messages']

    response = model.invoke(messages)

    return {'messages': [response]}

checkpointer = MemorySaver()

graph = StateGraph(ChatbotState)


chatode = graph.add_node('chatnode', chatnode)

chatedge = graph.add_edge(START, 'chatnode')
end_edge = graph.add_edge('chatnode', END)


chatbot = graph.compile(checkpointer=checkpointer)


thread_id = '1'
while True:
    user_message = input("type your query : ")

    print('User : ', user_message)

    if user_message.strip().lower() in ['exit', 'quit', 'bye']:
        break

    config = {'configurable' : {'thread_id' : thread_id}}

    response = chatbot.invoke({
    'messages' : [
        HumanMessage(content=user_message)
        ]
    }, config=config)

    print('AI : ', response['messages'][-1].content)

# initial_state= {
#     'messages' : [
#         HumanMessage(content='What is the winter capital of Karnataka?')
#     ]
# }


# result = chatbot.invoke(initial_state)

# print(result)