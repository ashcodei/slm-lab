"""Complete a checkpoint package with its exact own tokenizer, without overwrite."""
import os
from pathlib import Path
import tempfile

from tokenizers import Tokenizer

from .corpus import file_hash


def ensure(source, directory, *, expected_sha256, expected_vocab_size=32768):
    source,directory = Path(source),Path(directory)
    destination = directory/'tokenizer.json'
    if file_hash(source) != expected_sha256:
        raise ValueError('Exact own tokenizer source bytes required')
    tokenizer = Tokenizer.from_file(str(source))
    if (tokenizer.get_vocab_size() != expected_vocab_size
            or [tokenizer.token_to_id(t) for t in ['<pad>','<bos>','<eos>']] != [0,1,2]):
        raise ValueError('Own trained vocabulary and original PAD/BOS/EOS identities required')
    if destination.exists():
        if file_hash(destination) != expected_sha256:
            raise ValueError('An existing different tokenizer must not be overwritten')
        return dict(path=str(destination),sha256=expected_sha256,already_present=True)
    with tempfile.NamedTemporaryFile(dir=directory,prefix='.tokenizer-',suffix='.tmp',delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(source.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if file_hash(temporary) != expected_sha256:
            raise ValueError('Complete tokenizer copy must validate before publication')
        try:
            os.link(temporary,destination)
        except FileExistsError:
            if file_hash(destination) != expected_sha256:
                raise ValueError('Concurrent different package tokenizer must not be overwritten')
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(destination),sha256=expected_sha256,already_present=False)
