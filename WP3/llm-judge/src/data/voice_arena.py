from data import human_ai


def list_conversations():
    return human_ai.list_calls()


def get_input(convo_id, modality):
    if modality == "text":
        transcript = human_ai.get_call_transcript(convo_id)
        if transcript is None:
            raise FileNotFoundError(f"no transcript for call {convo_id} yet: run transcribe_voice_arena.py first")
        return transcript
    return human_ai.get_call_audio_mono_path(convo_id)


def get_label(convo_id):
    # Bradley-Terry strength (0-1) on whether the call accomplished the airline task, the most
    # direct analog to "success" for this task-oriented dataset. See human_ai.get_call_strengths.
    return human_ai.get_call_strengths("task_capability")[convo_id]


def get_category(convo_id):
    return human_ai.get_call_metadata(convo_id)["provider"]
