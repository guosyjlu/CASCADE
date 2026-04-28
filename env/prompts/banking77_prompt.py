CASE_PROMPT = """[Task] {task}
[Answer] \\boxed{{{answer}}}
"""

CASE_PROMPT_REFLEXION = """[Task] {task}
[Answer] \\boxed{{{answer}}}
[Feedback] {feedback}
"""


CBR_PROMPT = """Given a user query, your task is to route it to the right target place. Here are all the possible target places (splitted by comma):
[card arrival,card linking,exchange rate,card payment wrong exchange rate,extra charge on statement,pending cash withdrawal,fiat currency support,card delivery estimate,automatic top up,card not working,exchange via app,lost or stolen card,age limit,pin blocked,contactless not working,top up by bank transfer charge,pending top up,cancel transfer,top up limits,wrong amount of cash received,card payment fee charged,transfer not received by recipient,supported cards and currencies,getting virtual card,card acceptance,top up reverted,balance not updated after cheque or cash deposit,card payment not recognised,edit personal details,why verify identity,unable to verify identity,get physical card,visa or mastercard,topping up by card,disposable card limits,compromised card,atm support,direct debit payment not recognised,passcode forgotten,declined cash withdrawal,pending card payment,lost or stolen phone,request refund,declined transfer,Refund not showing up,declined card payment,pending transfer,terminate account,card swallowed,transaction charged twice,verify source of funds,transfer timing,reverted card payment?,change pin,beneficiary not allowed,transfer fee charged,receiving money,failed transfer,transfer into account,verify top up,getting spare card,top up by cash or cheque,order physical card,virtual card not working,wrong exchange rate for cash withdrawal,get disposable virtual card,top up failed,balance not updated after bank transfer,cash withdrawal not recognised,exchange charge,top up by card charge,activate my card,cash withdrawal charge,card about to expire,apple pay or google pay,verify my identity,country support]

Here are some relevant cases:
{case_prompt}

Now, given the user query:
{task}
Please directly provide the targeted routing place in the following format:
\\boxed{{place}}
"""


ZERO_SHOT_PROMPT = """Given a user query, your task is to route it to the right target place. Here are all the possible target places (splitted by comma):
[card arrival,card linking,exchange rate,card payment wrong exchange rate,extra charge on statement,pending cash withdrawal,fiat currency support,card delivery estimate,automatic top up,card not working,exchange via app,lost or stolen card,age limit,pin blocked,contactless not working,top up by bank transfer charge,pending top up,cancel transfer,top up limits,wrong amount of cash received,card payment fee charged,transfer not received by recipient,supported cards and currencies,getting virtual card,card acceptance,top up reverted,balance not updated after cheque or cash deposit,card payment not recognised,edit personal details,why verify identity,unable to verify identity,get physical card,visa or mastercard,topping up by card,disposable card limits,compromised card,atm support,direct debit payment not recognised,passcode forgotten,declined cash withdrawal,pending card payment,lost or stolen phone,request refund,declined transfer,Refund not showing up,declined card payment,pending transfer,terminate account,card swallowed,transaction charged twice,verify source of funds,transfer timing,reverted card payment?,change pin,beneficiary not allowed,transfer fee charged,receiving money,failed transfer,transfer into account,verify top up,getting spare card,top up by cash or cheque,order physical card,virtual card not working,wrong exchange rate for cash withdrawal,get disposable virtual card,top up failed,balance not updated after bank transfer,cash withdrawal not recognised,exchange charge,top up by card charge,activate my card,cash withdrawal charge,card about to expire,apple pay or google pay,verify my identity,country support]
Now, given the user query:
{task}
Please directly provide the targeted routing place in the following format:
\\boxed{{place}}
"""
