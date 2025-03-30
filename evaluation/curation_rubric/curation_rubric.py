from Kramer import get_all_certs, Curation
from Chain import Chain, Model, Prompt, Parser, ModelAsync, AsyncChain
from pathlib import Path
from pydantic import BaseModel, Field

dir_path = Path(__file__).parent
template_files = dir_path.glob("*.jinja2")
template_strings = [template_file.read_text() for template_file in template_files]


class CurationRubric(BaseModel):
    dimension: str = Field(description="The dimension of the rubric being evaluated")
    score: int = Field(ge=1, le=5, description="The score given to the dimension (1-5)")
    rationale: str = Field(description="A brief justification for the score given")


def evaluate_curation(
    curation: Curation, preferred_model: str = "claude", verbose=True
) -> tuple[list[CurationRubric], float]:
    """
    Evaluation function for curation.
    """
    # Load the template strings
    dimension_prompts = [
        Prompt(template_string) for template_string in template_strings
    ]
    curation_rubrics = []
    for index, dimension_prompt in enumerate(dimension_prompts):
        model = Model(preferred_model)
        parser = Parser(CurationRubric)  # type: ignore
        chain = Chain(prompt=dimension_prompt, model=model, parser=parser)
        response = chain.run(  # type: ignore
            input_variables={"snapshot": curation.snapshot},
            cache=False,
            verbose=verbose,
        )
        curation_rubrics.append(response.content)
    overall_score = 0
    for curation_rubric in curation_rubrics:
        dimension = curation_rubric.dimension
        score = curation_rubric.score
        rationale = curation_rubric.rationale
        overall_score += score
        final_score = overall_score / len(curation_rubrics)
    if not final_score:  # type: ignore
        raise ValueError("Unable to generate score for some reason.")
    else:
        return curation_rubrics, final_score


def evaluate_curation_async(
    curation: Curation, preferred_model: str = "claude", verbose=True
) -> tuple[list[CurationRubric], float]:
    """
    Evaluation function for curation.
    """
    # Load the template strings
    dimension_prompts = [
        Prompt(template_string) for template_string in template_strings
    ]
    # Render the prompts with the input variables
    prompt_strings = [
        prompt.render(input_variables={"snapshot": curation.snapshot})
        for prompt in dimension_prompts
    ]
    # Run our list of prompt strings through async
    model = ModelAsync(preferred_model)
    parser = Parser(CurationRubric)  # type: ignore
    chain = AsyncChain(model=model, parser=parser)
    responses = chain.run(
        prompt_strings=prompt_strings, verbose=verbose, cache=False
    )  # Type: ignore
    # Process the curation_rubrics
    curation_rubrics = [response.content for response in responses]
    overall_score = 0
    for curation_rubric in curation_rubrics:
        dimension = curation_rubric.dimension
        score = curation_rubric.score
        rationale = curation_rubric.rationale
        overall_score += score
    final_score = overall_score / len(curation_rubrics)
    if not final_score:  # type: ignore
        raise ValueError("Unable to generate score for some reason.")
    else:
        return curation_rubrics, final_score


if __name__ == "__main__":
    certs = get_all_certs()
    for index, cert in enumerate(certs):
        curation_rubrics, final_score = evaluate_curation_async(
            cert, preferred_model="gpt-4o", verbose=False
        )
        print("----------------------------")
        print(cert.title)
        print(f"Final Score: {final_score:.2f}")
        print("----------------------------")
        for curation_rubric in curation_rubrics:
            dimension = curation_rubric.dimension
            score = curation_rubric.score
            rationale = curation_rubric.rationale
            print(f"Dimension: {dimension}")
            print(f"Score: {score}")
            print(f"Rationale: {rationale}")
            print("----------------------------")
