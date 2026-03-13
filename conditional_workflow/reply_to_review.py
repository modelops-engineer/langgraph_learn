from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional, Annotated
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv



load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')
parser = StrOutputParser()

class sentimentSchema(BaseModel):
    sentiment : Literal['positive','neutral','negative'] = Field(description='It is the sentiment of the review given by the user')


class diagnosisSchema(BaseModel):
    issue_type: Literal["ui","ux","hardware","software","network","other"] = Field(description='It is the category of the issue mentioned in the review')
    custom_issue_type: Optional[str] = Field(
        default=None,
        description="A new category if issue_type is 'other'"
    )
    tone: Literal["neutral","angry","frustrated","happy"] = Field(description="Emotional tone of the user review")
    urgency: Literal["low","medium","high"] = Field(description='How urgent or critical the issue is')

sentiment_check_model = model.with_structured_output(sentimentSchema)
diagnosis_model = model.with_structured_output(diagnosisSchema)


class ReplyState(TypedDict):
    review : str
    sentiment : str
    diagnosis : dict
    reply : str

graph = StateGraph(ReplyState)


def find_sentiment(state : ReplyState) -> ReplyState:
    prompt = f"""You are sentiment analyser for the review given for phones. Find the sentiment of the review by the user but do not get overwhelmed with words present in the review, understand the review to get to final conclusion,\n
    review : {state['review']}
    """

    sentiment = sentiment_check_model.invoke(prompt).sentiment

    return {'sentiment': sentiment}

def positive_response(state : ReplyState) -> ReplyState:
    prompt = f"""
        generate a thank you response for the user for the review.\n
        Review : {state['review']}\n
        And if the name of the user is present then greet them with their name, but do not get confuse with other names provided in the review. If you are not sure do not take any name.
"""
    
    response = model.invoke(prompt).content

    return {'reply':response}

def run_diagnosis(state : ReplyState) -> ReplyState:
    prompt = f"""
        As per the review of the user run diagnosis on the review to find issue type, tone, urgency of the user.\n
        Review : {state['review']}\n
"""
    diagnosis = diagnosis_model.invoke(prompt).model_dump()

    return {'diagnosis':diagnosis}

def negative_response(state : ReplyState) -> ReplyState:

    diagnosis = state['diagnosis']
    prompt = f"""
        The user had a '{diagnosis['issue_type']}' or '{diagnosis['custom_issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
        Write an empathetic, helpful message message to the user over the review. \n 
        review : {state['review']}
"""
    response = model.invoke(prompt)

    return {'reply':response}


def sentiment_check(state:ReplyState):
    if state['sentiment'] == 'positive' or state['sentiment'] == 'neutral':
        return 'positive_response'
    else:
        return 'run_diagnosis'


# node
node1 = graph.add_node('find_sentiment', find_sentiment)
node2 = graph.add_node('positive_response', positive_response)
node3 = graph.add_node('run_diagnosis', run_diagnosis)
node4 = graph.add_node('negative_response', negative_response)

# add edge
edge1 = graph.add_edge(START, 'find_sentiment')
edge2 = graph.add_conditional_edges('find_sentiment', sentiment_check)
edge3 = graph.add_edge('positive_response', END)
edge4 = graph.add_edge('run_diagnosis', 'negative_response')
edge5 = graph.add_edge('negative_response', END)


# compile
workflow = graph.compile()

# result
initial_state = {
    'review' : "I've been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}

result = workflow.invoke(initial_state)

print(result)