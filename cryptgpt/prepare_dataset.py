# coding: utf-8
from functools import partial
import multiprocessing
import os
import secrets

from datasets import load_dataset
from transformers import AutoTokenizer
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# Retrieve the encryption key from the environment
key = os.environ["ENCRYPTION_KEY"].encode()  # Ensure the key is in bytes
nonce = secrets.token_bytes(16)  # Generate a random nonce for ChaCha20

num_proc = multiprocessing.cpu_count()
num_proc_load_dataset = num_proc // 2

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def chacha20_encrypt(data, key, nonce):
    algorithm = algorithms.ChaCha20(key, nonce)
    cipher = Cipher(algorithm, mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(data)

def chacha20_decrypt(data, key, nonce):
    algorithm = algorithms.ChaCha20(key, nonce)
    cipher = Cipher(algorithm, mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(data)

def encrypt(message, key, nonce):
    data = message.encode('utf-8')
    encrypted_data = chacha20_encrypt(data, key, nonce)
    return encrypted_data.hex()

def decrypt(cipher_hex, key, nonce):
    encrypted_data = bytes.fromhex(cipher_hex)
    decrypted_data = chacha20_decrypt(encrypted_data, key, nonce)
    return decrypted_data.decode('utf-8')

def test_encrypt_decrypt():
    message = "i love peanuts"
    key = os.environ["ENCRYPTION_KEY"].encode()
    nonce = secrets.token_bytes(16)
    encrypted_message = encrypt(message, key, nonce)
    decrypted_message = decrypt(encrypted_message, key, nonce)

    print("Original message: " + message)
    print("Encrypted message (hex): " + encrypted_message)
    print("Decrypted message: " + decrypted_message)

    assert decrypted_message == message

encrypt_ = partial(encrypt, key=key, nonce=nonce)
decrypt_ = partial(decrypt, key=key, nonce=nonce)

if __name__ == "__main__":
    test_encrypt_decrypt()

    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    dataset = dataset.map(
        lambda row: dict(encrypted=encrypt_(row["text"]) + gpt2_tokenizer.eos_token),
        num_proc=num_proc-2
    )

    dataset = dataset.remove_columns(["text"]).rename_column("encrypted", "text")

    dataset.push_to_hub("diwank/encrypted-openwebtext", num_shards={"train": 20}, private=True)

    # Save the nonce for future decryption
    with open("chacha20_nonce.bin", "wb") as nonce_file:
        nonce_file.write(nonce)
