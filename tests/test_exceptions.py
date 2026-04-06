from slm.exceptions import (
    SLMError,
    ConfigError,
    IngestError,
    EmbeddingError,
    StoreError,
    RetrievalError,
    RewriteError,
)


def test_all_exceptions_inherit_from_slm_error():
    for exc_class in [ConfigError, IngestError, EmbeddingError, StoreError, RetrievalError, RewriteError]:
        err = exc_class("test message")
        assert isinstance(err, SLMError)
        assert isinstance(err, Exception)
        assert str(err) == "test message"


def test_slm_error_is_base():
    err = SLMError("base error")
    assert str(err) == "base error"


def test_exceptions_are_distinct():
    classes = [ConfigError, IngestError, EmbeddingError, StoreError, RetrievalError, RewriteError]
    for i, cls_a in enumerate(classes):
        for cls_b in classes[i + 1:]:
            assert not issubclass(cls_a, cls_b)
            assert not issubclass(cls_b, cls_a)
