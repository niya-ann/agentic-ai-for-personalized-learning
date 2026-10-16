"""
Curriculum & Planning Agent.

Initial planning uses the Knowledge Graph to topologically order the
learner's unmet concepts.

Adaptive replanning performs MINIMAL PLAN REPAIR:
- starts from the learner's existing active plan,
- preserves the existing plan contents,
- identifies detected learning-gap concepts,
- moves gap concepts earlier only when prerequisite constraints allow,
- never rebuilds the entire course curriculum during a replan.
"""

from app import kg as kg_mod

MASTERY_THRESHOLD = 0.7  # a concept counts as "known" once mastery >= this


def build_plan(profile, g):
    """
    Build the learner's initial learning plan.

    This is GLOBAL planning:
    - retrieve all concepts belonging to the learner's course,
    - remove concepts already mastered,
    - topologically order the remaining concepts using the Knowledge Graph.
    """
    course_id = profile["goal_course"]

    all_concepts = kg_mod.get_concepts_for_course(g, course_id)

    mastery = profile["mastery"]

    unmet = [
        c
        for c in all_concepts
        if mastery.get(c, 0.0) < MASTERY_THRESHOLD
    ]

    ordered = kg_mod.topological_concept_order(g, unmet)

    return ordered, course_id


def build_replan_with_gap_priority(
    profile,
    g,
    gap_concept_ids,
    active_plan
):
    """
    Perform MINIMAL ADAPTIVE PLAN REPAIR.

    Unlike build_plan(), this function does NOT rebuild the entire
    unmet curriculum.

    Instead:
    1. Start from the learner's currently active plan.
    2. Preserve all concepts already approved in that plan.
    3. Identify concepts that are currently learning gaps.
    4. Pull those gap concepts forward as early as their prerequisites allow.
    5. Return the repaired plan.

    This means that a failed assessment on one concept does not suddenly
    reintroduce unrelated concepts that were not part of the learner's
    current active plan.
    """

    course_id = profile["goal_course"]

    # If there is no active plan, fall back safely to normal planning.
    if not active_plan:
        return build_plan(profile, g)

    # Copy the existing plan so that we never mutate the DB object directly.
    current_order = list(active_plan.get("concept_order", []))

    # Remove any duplicate concept IDs while preserving order.
    repaired_order = []
    seen = set()

    for concept_id in current_order:
        if concept_id not in seen:
            repaired_order.append(concept_id)
            seen.add(concept_id)

    # Only gaps that are actually present in the active plan should
    # participate in local repair.
    gap_set = {
        concept_id
        for concept_id in gap_concept_ids
        if concept_id in repaired_order
    }

    # If the gap is not part of the current active plan, do not silently
    # expand the learner's plan with the entire course curriculum.
    #
    # We preserve the current plan and let the orchestrator/explanation
    # layer record that the detected gap was outside the active plan.
    if not gap_set:
        return repaired_order, course_id

    # Move each gap concept as early as possible while preserving all
    # prerequisite relationships that are represented in the active plan.
    repaired_order = _pull_forward_respecting_prereqs(
        repaired_order,
        gap_set,
        g
    )

    return repaired_order, course_id


def _pull_forward_respecting_prereqs(order, priority_set, g):
    """
    Move priority/gap concepts earlier in the existing plan, but never
    move a concept ahead of one of its prerequisites.

    Unlike the original implementation, this checks the full prerequisite
    ancestry rather than only checking the immediately preceding concept.
    """

    prereq_g = kg_mod.get_prereq_subgraph(g)

    result = list(order)

    for priority_concept in priority_set:

        if priority_concept not in result:
            continue

        current_index = result.index(priority_concept)

        # Find every prerequisite ancestor of the gap concept.
        ancestors = nx_ancestors(prereq_g, priority_concept)

        # Only prerequisites that are actually present in this active plan
        # matter for the ordering of this particular plan.
        relevant_prereqs = [
            concept_id
            for concept_id in result
            if concept_id in ancestors
        ]

        if relevant_prereqs:
            # The gap must remain AFTER its latest prerequisite.
            latest_prereq_index = max(
                result.index(concept_id)
                for concept_id in relevant_prereqs
            )

            target_index = latest_prereq_index + 1
        else:
            # No prerequisite of the gap exists in the active plan.
            # Therefore it can be moved to the beginning.
            target_index = 0

        # Do nothing if it is already at the earliest valid position.
        if target_index >= current_index:
            continue

        # Remove the concept from its current position.
        result.pop(current_index)

        # Removing an earlier element changes indices, so calculate the
        # insertion point again against the updated list.
        if target_index > len(result):
            target_index = len(result)

        result.insert(target_index, priority_concept)

    return result


def nx_ancestors(graph, node):
    """
    Small local helper to obtain all prerequisite ancestors without
    requiring the planner to import NetworkX directly.

    The Knowledge Graph is already represented as a NetworkX DiGraph.
    """
    ancestors = set()

    stack = list(graph.predecessors(node))

    while stack:
        parent = stack.pop()

        if parent in ancestors:
            continue

        ancestors.add(parent)

        for predecessor in graph.predecessors(parent):
            if predecessor not in ancestors:
                stack.append(predecessor)

    return ancestors