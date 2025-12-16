from mistralai import Mistral
import os


with Mistral(
    api_key="lDjD4RVO8FRtHiCCALGpdRm5uhEsqiiO",
) as mistral:

    res = mistral.models.list()

    # Handle response
    print(res)

