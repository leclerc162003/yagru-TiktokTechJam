from starter.agent import Agent


def test_preference_memory_evolves():
    agent = Agent()

    profile = {
        "user_id": "memory_test_user",
        "preference_tags": ["comfort", "durability"],
    }

    # Build leather preference.
    for i in range(3):
        session_id = f"leather_{i}"

        agent.reset(session_id, profile)

        agent.respond(
            session_id,
            "I'm looking for shoes. "
            "A key requirement is: Material: leather.",
            1,
            10,
        )

    key = agent._memory_key(
        "leather_2",
        profile,
    )

    memory = agent._load_learned_preferences(key)

    assert memory == [
        {
            "type": "material",
            "value": "leather",
            "score": 3.0,
        }
    ]

    # User's preference changes.
    for i in range(3):
        session_id = f"canvas_{i}"

        agent.reset(session_id, profile)

        agent.respond(
            session_id,
            "I'm looking for shoes. "
            "A key requirement is: Material: canvas.",
            1,
            10,
        )

    memory = agent._load_learned_preferences(key)

    assert memory == [
        {
            "type": "material",
            "value": "canvas",
            "score": 3.0,
        }
    ]