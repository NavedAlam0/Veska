"""Tests for audio media processing."""

import base64

from veska import Agent
from veska.media import Audio, process_attachments
from veska.providers.claude_provider import ClaudeProvider, _to_claude_content_blocks
from veska.providers.base import BaseProvider, Message, ProviderResponse, ThinkingConfig
from veska.providers.openai_provider import OpenAIProvider, _to_openai_content_blocks


class RecordingProvider(BaseProvider):
    def __init__(self, provider_name: str, model: str, supports_audio: bool = False):
        super().__init__(api_key="test-key", model=model)
        self._provider_name = provider_name
        self._supports_audio = supports_audio
        self.calls = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        thinking: ThinkingConfig | None = None,
        stream: bool = False,
        **kwargs,
    ) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(content="done", model=self.model)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def supports_thinking(self) -> bool:
        return False

    def supports_audio_input(self) -> bool:
        return self._supports_audio

    def supported_audio_formats(self) -> set[str]:
        return {"wav", "mp3"} if self._supports_audio else set()


def test_process_audio_file_as_base64_block(tmp_path):
    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"fake audio")

    blocks = process_attachments([Audio(str(audio_path), language="en")])

    assert blocks == [
        {
            "type": "audio",
            "source_type": "base64",
            "data": base64.b64encode(b"fake audio").decode("utf-8"),
            "format": "mp3",
            "media_type": "audio/mpeg",
            "language": "en",
            "filename": "voice.mp3",
        }
    ]


def test_process_audio_content_as_base64_block():
    blocks = process_attachments([Audio(content=b"fake wav", format="wav")])

    assert blocks[0]["type"] == "audio"
    assert blocks[0]["source_type"] == "base64"
    assert blocks[0]["data"] == base64.b64encode(b"fake wav").decode("utf-8")
    assert blocks[0]["format"] == "wav"
    assert blocks[0]["media_type"] == "audio/wav"


def test_process_string_detects_audio_file(tmp_path):
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"fake wav")

    blocks = process_attachments([str(audio_path)])

    assert blocks[0]["type"] == "audio"
    assert blocks[0]["format"] == "wav"


def test_openai_formats_supported_audio_input():
    encoded = base64.b64encode(b"fake wav").decode("utf-8")
    blocks = [{
        "type": "audio",
        "source_type": "base64",
        "data": encoded,
        "format": "wav",
        "media_type": "audio/wav",
    }]

    openai_blocks = _to_openai_content_blocks(
        blocks,
        supports_audio_input=True,
        supported_audio_formats={"wav", "mp3"},
    )

    assert openai_blocks == [
        {
            "type": "input_audio",
            "input_audio": {
                "data": encoded,
                "format": "wav",
            },
        }
    ]


def test_openai_falls_back_for_unsupported_audio_format():
    blocks = [{
        "type": "audio",
        "source_type": "base64",
        "data": "abc",
        "format": "m4a",
        "media_type": "audio/mp4",
        "filename": "voice.m4a",
    }]

    openai_blocks = _to_openai_content_blocks(
        blocks,
        supports_audio_input=True,
        supported_audio_formats={"wav", "mp3"},
    )

    assert openai_blocks[0]["type"] == "text"
    assert "does not accept this audio input directly" in openai_blocks[0]["text"]


def test_openai_text_model_falls_back_for_audio_input():
    encoded = base64.b64encode(b"fake wav").decode("utf-8")
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
    messages = [
        Message(role="user", content=[{
            "type": "audio",
            "source_type": "base64",
            "data": encoded,
            "format": "wav",
            "media_type": "audio/wav",
            "filename": "voice.wav",
        }])
    ]

    api_kwargs = provider._build_api_kwargs(messages, tools=None)

    content = api_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "selected OpenAI model does not accept" in content[0]["text"]


def test_openai_audio_model_sends_audio_to_same_provider_model():
    encoded = base64.b64encode(b"fake wav").decode("utf-8")
    provider = OpenAIProvider(api_key="test-key", model="gpt-audio")
    messages = [
        Message(role="user", content=[{
            "type": "audio",
            "source_type": "base64",
            "data": encoded,
            "format": "wav",
            "media_type": "audio/wav",
        }])
    ]

    api_kwargs = provider._build_api_kwargs(messages, tools=None)

    content = api_kwargs["messages"][0]["content"]
    assert api_kwargs["model"] == "gpt-audio"
    assert content == [{
        "type": "input_audio",
        "input_audio": {
            "data": encoded,
            "format": "wav",
        },
    }]


def test_provider_audio_capabilities_are_provider_specific():
    openai_text = OpenAIProvider(api_key="test-key", model="gpt-4o")
    openai_audio = OpenAIProvider(api_key="test-key", model="gpt-audio")
    claude = ClaudeProvider(api_key="test-key", model="claude-sonnet-4-6")

    assert not openai_text.supports_audio_input()
    assert openai_text.supported_audio_formats() == set()
    assert openai_audio.supports_audio_input()
    assert openai_audio.supported_audio_formats() == {"wav", "mp3"}
    assert not claude.supports_audio_input()
    assert claude.supported_audio_formats() == set()


def test_claude_formats_audio_as_clear_fallback_text():
    blocks = [{
        "type": "audio",
        "source_type": "base64",
        "data": "abc",
        "format": "mp3",
        "media_type": "audio/mpeg",
        "filename": "voice.mp3",
    }]

    claude_blocks = _to_claude_content_blocks(blocks)

    assert claude_blocks[0]["type"] == "text"
    assert "does not currently send raw audio" in claude_blocks[0]["text"]


def test_agent_errors_before_calling_provider_when_audio_is_not_supported(tmp_path):
    audio_path = tmp_path / "voice.mp3"
    audio_path.write_bytes(b"fake audio")
    provider = RecordingProvider("claude", "claude-sonnet-4-6", supports_audio=False)
    agent = Agent(name="assistant", provider=provider)

    result = agent.run("Transcribe this", attachments=[Audio(str(audio_path))])

    assert not result.success
    assert provider.calls == 0
    assert "Audio input is not supported by provider 'claude'" in result.error


def test_agent_errors_before_calling_provider_when_audio_format_is_not_supported(tmp_path):
    audio_path = tmp_path / "voice.m4a"
    audio_path.write_bytes(b"fake audio")
    provider = RecordingProvider("openai", "gpt-audio", supports_audio=True)
    agent = Agent(name="assistant", provider=provider)

    result = agent.run("Transcribe this", attachments=[Audio(str(audio_path))])

    assert not result.success
    assert provider.calls == 0
    assert "format 'm4a' is not supported" in result.error
