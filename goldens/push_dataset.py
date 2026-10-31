import langfuse
from langfuse.api.commons.errors.not_found_error import NotFoundError

from evals.harness import load_goldens_from_local


def push_retriever_dataset(name: str, data_path: str):
    client = langfuse.Langfuse()

    try:
        client.get_dataset(name)
    except NotFoundError:
        client.create_dataset(name=name)

    # Load all items from the JSON golden file
    data = load_goldens_from_local(path=data_path)

    print(f"Syncing {len(data)} items to Langfuse dataset '{name}'...")

    # Iterate and upsert each record into the dataset
    for item in data:
        client.create_dataset_item(
            dataset_name=name,
            # Static ID pass karne se same ID par data update hoga, duplicate entries nahi banengi
            id=item.get("id"),
            input={
                "query": item.get("query")
            },
            expected_output={
                "ideal_answer": item.get("ideal_answer"),
            },
            metadata={
                "source": item.get("source")
            }
        )

    print(f"Successfully synced {len(data)} items to dataset '{name}' without duplicates.")




def push_generator_dataset(name: str, data_path: str):
    client = langfuse.Langfuse()

    try:
        client.get_dataset(name)
    except NotFoundError:
        client.create_dataset(name=name)

    # Load all items from the JSON golden file
    data = load_goldens_from_local(path=data_path)

    print(f"Syncing {len(data)} items to Langfuse dataset '{name}'...")

    # Iterate and upsert each record into the dataset
    for item in data:
        client.create_dataset_item(
            dataset_name=name,
            # Static ID pass karne se same ID par data update hoga, duplicate entries nahi banengi
            id=item.get("id"),
            input={
                "query": item.get("query")
            },
            expected_output={
                "ideal_context": item.get("ideal_context")
            },
            metadata={
                "source": item.get("source")
            }
        )

    print(f"Successfully synced {len(data)} items to dataset '{name}' without duplicates.")






def push_application_dataset(name: str, data_path: str):
    client = langfuse.Langfuse()

    try:
        client.get_dataset(name)
    except NotFoundError:
        client.create_dataset(name=name)

    # Load all items from the JSON golden file
    data = load_goldens_from_local(path=data_path)

    print(f"Syncing {len(data)} items to Langfuse dataset '{name}'...")

    # Iterate and upsert each record into the dataset
    for item in data:
        client.create_dataset_item(
            dataset_name=name,
            # Static ID pass karne se same ID par data update hoga, duplicate entries nahi banengi
            id=item.get("id"),
            input={
                "question": item.get("question")
            },
            expected_output={
                "ideal_answer": item.get("ideal_answer"),
            },
            metadata={
                "source": item.get("source")
            }
        )

    print(f"Successfully synced {len(data)} items to dataset '{name}' without duplicates.")


if __name__=="__main__":
    print("pushing retriever dataset")
    push_retriever_dataset(name="Retriever_Eval",data_path="goldens/eval_retriever.json")

    print("pushing generator dataset")
    push_retriever_dataset(name="Generator_Eval",data_path="goldens/eval_generator.json")

    print("pushing application dataset")
    push_retriever_dataset(name="Application_Eval",data_path="goldens/eval_application.json")
