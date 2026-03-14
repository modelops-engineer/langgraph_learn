from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated
from pydantic import Field, BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import operator


load_dotenv()

generator_model = ChatOpenAI(model='gpt-4o')
evaluator_model = ChatOpenAI(model='gpt-4o-mini')
optimizer_model = ChatOpenAI(model='gpt-4o')
parser = StrOutputParser()

class Evaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(description="Final evaluation result.")
    feedback: str = Field(description="feedback for the post.")

structured_evaluator_model = evaluator_model.with_structured_output(Evaluation)


class PostState(TypedDict):

    topic : str
    post : str
    evaluation : Literal['approved', 'improvement']
    feedback : str
    iteration : int
    max_interation : int

    post_gen_history : Annotated[list[str], operator.add]
    feedback_history : Annotated[list[str], operator.add]

def generate_post(state : PostState) -> PostState:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a funny and clever influencer. You have been posting on social from last 5 years and you post about recent developments in the world, day to day work, learnings. Basically you are digital content creator"),
            HumanMessage(content="""
            Write a short, original, and hilarious post on the topic: \n {topic}.

            Rules:
            - Do NOT use question-answer format.
            - Max 280 characters.
            - Use observational humor, irony, sarcasm, or cultural references.
            - Think in meme logic, punchlines, or relatable takes.
            - Use simple, day to day english
""")
        ]
    )

    chain = prompt | generator_model | parser

    response = chain.invoke({'topic' : state['topic']})

    return {'post':response, 'post_gen_history':[response]}

def evaluate_post(state : PostState) -> PostState:
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate posts based on humor, originality, virality, and post format."),
    HumanMessage(content=f"""
        Evaluate the following post:

        post: "{state['post']}"

        Use the criteria below to evaluate the post:

        1. Originality - Is this fresh, or have you seen it a hundred times before?  
        2. Humor - Did it genuinely make you smile, laugh, or chuckle?  
        3. Punchiness - Is it short, sharp, and scroll-stopping?  
        4. Virality Potential - Would people repost or share it?  
        5. Format - Is it a well-formed post (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

        Auto-reject if:
        - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
        - It exceeds 280 characters
        - It reads like a traditional setup-punchline joke
        - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

        ### Respond ONLY in structured format:
        - evaluation: "approved" or "needs_improvement"  
        - feedback: One paragraph explaining the strengths and weaknesses 
        """)
    ]

    response = structured_evaluator_model.invoke(messages)

    return {'evaluation': response.evaluation, 'feedback':response.feedback, 'feedback_history':[response.feedback]}
    

def optimize_post(state : PostState) -> PostState:
    messages = [
        SystemMessage(content="You punch up posts for virality and humor based on given feedback."),
        HumanMessage(content=f"""
        Improve the post based on this feedback:
        "{state['feedback']}"

        Topic: "{state['topic']}"
        Original post:
        {state['post']}

        Re-write it as a short, viral-worthy post. Avoid Q&A style and stay under 280 characters.
        """)
    ]

    response = optimizer_model.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'post':response, 'post_gen_history':[response], 'iteration': iteration}

def route(state : PostState):
    if state['evaluation'] == 'approved' or state['iteration'] > state['max_iteration']:
        return 'approved'
    else:
        return 'improvement'


graph = StateGraph(PostState)

# add node
node1 = graph.add_node('generate', generate_post)
node2 = graph.add_node('evaluate', evaluate_post)
node3 = graph.add_node('optimize', optimize_post)


# add edge
edge1 = graph.add_edge(START, 'generate')
edge2 = graph.add_edge('generate', 'evaluate')
edge3 = graph.add_conditional_edges('evaluate', route, {'approved' : END, 'improvement':'optimize'})
edge4 = graph.add_edge('optimize', 'evaluate')


workflow = graph.compile()

initial_state = {
    'topic' : 'AI is not the future',
    'iteration' : 0,
    'max_iteration' : 10
}

result = workflow.invoke(initial_state)

print(result)