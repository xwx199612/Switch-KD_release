import asyncio
from pathlib import Path

from app.vlm_distill.docker_service import (
    app,
    lifespan,
    create_transition_inferencer,
)
from app.vlm_distill.state_registry import StateRegistry


BEFORE = Path("samples/transition/before.jpg")
AFTER = Path("samples/transition/after.jpg")


async def main():
    assert BEFORE.exists(), BEFORE
    assert AFTER.exists(), AFTER

    before_bytes = BEFORE.read_bytes()
    after_bytes = AFTER.read_bytes()

    print("before bytes:", len(before_bytes))
    print("after bytes :", len(after_bytes))

    async with lifespan(app):
        context = app.state.runtime_context

        print("model_instance_id    :", context.model_instance_id)
        print("processor_instance_id:", context.processor_instance_id)
        print("model_load_count      :", context.model_load_count)

        inferencer = create_transition_inferencer(context)

        registry = StateRegistry(
            transition_inferencer=inferencer
        )

        result = registry.resolve_images(
            before_bytes,
            after_bytes,
        )

        print("\n=== STATE RESOLUTION ===")
        print(
            result.before.state_id,
            "->",
            result.after.state_id
        )

        print("\nbefore:")
        print("  state_id:", result.before.state_id)
        print("  is_new  :", result.before.is_new)
        print("  score   :", result.before.score)

        print("\nafter:")
        print("  state_id:", result.after.state_id)
        print("  is_new  :", result.after.is_new)
        print("  score   :", result.after.score)

        print("\nregistry size:", len(registry))

        print("\n=== REGISTERED STATES ===")
        for state_id, registered in registry.states.items():
            print(state_id)
            print("  elements       :", sorted(registered.fingerprint.elements))
            print("  focus_context  :", registered.fingerprint.focus_context)
            print("  focused_element:", registered.fingerprint.focused_element)


if __name__ == "__main__":
    asyncio.run(main())
