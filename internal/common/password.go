//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package common

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/asn1"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"golang.org/x/crypto/pbkdf2"
	"golang.org/x/crypto/scrypt"
)

// CheckWerkzeugPassword verifies a password against a werkzeug password hash
// Supports both pbkdf2 and scrypt formats
func CheckWerkzeugPassword(password, hashStr string) bool {
	if strings.HasPrefix(hashStr, "scrypt:") {
		return checkScryptPassword(password, hashStr)
	}
	if strings.HasPrefix(hashStr, "pbkdf2:") {
		return checkPBKDF2Password(password, hashStr)
	}
	return false
}

// checkScryptPassword verifies password using scrypt format
// Format: scrypt:n:r:p$base64(salt)$hex(hash)
// IMPORTANT: werkzeug uses the base64-encoded salt string as UTF-8 bytes, NOT the decoded bytes
func checkScryptPassword(password, hashStr string) bool {
	parts := strings.Split(hashStr, "$")
	if len(parts) != 3 {
		return false
	}

	params := strings.Split(parts[0], ":")
	if len(params) != 4 || params[0] != "scrypt" {
		return false
	}

	n, err := strconv.ParseInt(params[1], 10, 0)
	if err != nil || n <= 0 {
		return false
	}
	r, err := strconv.ParseInt(params[2], 10, 0)
	if err != nil || r <= 0 {
		return false
	}
	p, err := strconv.ParseInt(params[3], 10, 0)
	if err != nil || p <= 0 {
		return false
	}

	saltB64 := parts[1]
	hashHex := parts[2]

	// IMPORTANT: werkzeug uses the base64 string as UTF-8 bytes, NOT decoded bytes
	// This is the key difference from standard implementations
	salt := []byte(saltB64)

	// Decode hash from hex
	expectedHash, err := hex.DecodeString(hashHex)
	if err != nil {
		return false
	}

	computed, err := scrypt.Key([]byte(password), salt, int(n), int(r), int(p), len(expectedHash))
	if err != nil {
		return false
	}

	return constantTimeCompare(expectedHash, computed)
}

// checkPBKDF2Password verifies password using PBKDF2 format
// Format: pbkdf2:sha256:iterations$base64(salt)$base64(hash)
func checkPBKDF2Password(password, hashStr string) bool {
	parts := strings.Split(hashStr, "$")
	if len(parts) != 3 {
		return false
	}

	methodParts := strings.Split(parts[0], ":")
	if len(methodParts) != 3 || methodParts[0] != "pbkdf2" {
		return false
	}

	iterations, err := strconv.Atoi(methodParts[2])
	if err != nil {
		return false
	}

	salt := parts[1]
	expectedHash := parts[2]

	saltBytes, err := base64.StdEncoding.DecodeString(salt)
	if err != nil {
		saltBytes, err = hex.DecodeString(salt)
		if err != nil {
			return false
		}
	}

	key := pbkdf2.Key([]byte(password), saltBytes, iterations, 32, sha256.New)
	computedHash := base64.StdEncoding.EncodeToString(key)

	return computedHash == expectedHash
}

// constantTimeCompare performs constant time comparison
func constantTimeCompare(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	var result byte
	for i := 0; i < len(a); i++ {
		result |= a[i] ^ b[i]
	}
	return result == 0
}

// IsWerkzeugHash checks if a hash is in werkzeug format
func IsWerkzeugHash(hashStr string) bool {
	return strings.HasPrefix(hashStr, "scrypt:") || strings.HasPrefix(hashStr, "pbkdf2:")
}

// GenerateWerkzeugPasswordHash generates a werkzeug-compatible password hash using scrypt
// This matches Python werkzeug's default behavior
func GenerateWerkzeugPasswordHash(password string) (string, error) {
	// Generate random bytes (12 bytes will produce 16-char base64 string)
	randomBytes := make([]byte, 12)
	if _, err := rand.Read(randomBytes); err != nil {
		return "", err
	}

	// Encode to base64 string (this will be 16 characters)
	saltB64 := base64.StdEncoding.EncodeToString(randomBytes)

	// Use scrypt with werkzeug default parameters: N=32768, r=8, p=1, keyLen=64
	// IMPORTANT: werkzeug uses the base64 string as UTF-8 bytes, NOT the decoded bytes
	hash, err := scrypt.Key([]byte(password), []byte(saltB64), 32768, 8, 1, 64)
	if err != nil {
		return "", err
	}

	// Format: scrypt:n:r:p$base64(salt)$hex(hash)
	return fmt.Sprintf("scrypt:32768:8:1$%s$%x", saltB64, hash), nil
}

// DecryptPassword decrypts the password using RSA private key
// The password is expected to be base64 encoded RSA encrypted data
// If decryption fails, the original password is returned (assumed to be plain text)
func DecryptPassword(encryptedPassword string) (string, error) {
	// Try to decode base64
	ciphertext, err := base64.StdEncoding.DecodeString(encryptedPassword)
	if err != nil {
		// If base64 decoding fails, assume it's already a plain password
		return encryptedPassword, nil
	}

	// Load private key
	privateKey, err := LoadPrivateKey()
	if err != nil {
		return "", err
	}

	// Decrypt using PKCS#1 v1.5
	plaintext, err := rsa.DecryptPKCS1v15(nil, privateKey, ciphertext)
	if err != nil {
		// If decryption fails, assume it's already a plain password
		return encryptedPassword, nil
	}

	return string(plaintext), nil
}

// LoadPrivateKey loads and decrypts the RSA private key from conf/private.pem
func LoadPrivateKey() (*rsa.PrivateKey, error) {
	keyData, err := os.ReadFile("conf/private.pem")
	if err != nil {
		return nil, fmt.Errorf("failed to read private key file: %w", err)
	}

	block, _ := pem.Decode(keyData)
	if block == nil {
		return nil, errors.New("failed to decode PEM block")
	}

	var keyDER []byte
	switch {
	case block.Type == "ENCRYPTED PRIVATE KEY":
		keyDER, err = decryptPKCS8PrivateKey(block.Bytes, []byte("Welcome"))
	case block.Headers["Proc-Type"] == "4,ENCRYPTED":
		keyDER, err = x509.DecryptPEMBlock(block, []byte("Welcome"))
	default:
		keyDER = block.Bytes
	}
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt private key: %w", err)
	}

	if privateKey, parseErr := x509.ParsePKCS1PrivateKey(keyDER); parseErr == nil {
		return privateKey, nil
	}
	privateKey, err := x509.ParsePKCS8PrivateKey(keyDER)
	if err != nil {
		return nil, fmt.Errorf("failed to parse private key: %w", err)
	}
	rsaPrivateKey, ok := privateKey.(*rsa.PrivateKey)
	if !ok {
		return nil, errors.New("private key is not RSA")
	}
	return rsaPrivateKey, nil
}

type pkcs8AlgorithmIdentifier struct {
	Algorithm  asn1.ObjectIdentifier
	Parameters asn1.RawValue `asn1:"optional"`
}

type encryptedPKCS8PrivateKey struct {
	EncryptionAlgorithm pkcs8AlgorithmIdentifier
	EncryptedData       []byte
}

type pbes2Parameters struct {
	KeyDerivationFunc pkcs8AlgorithmIdentifier
	EncryptionScheme  pkcs8AlgorithmIdentifier
}

type pbkdf2Parameters struct {
	Salt           asn1.RawValue
	IterationCount int
	KeyLength      int                      `asn1:"optional"`
	PRF            pkcs8AlgorithmIdentifier `asn1:"optional"`
}

func decryptPKCS8PrivateKey(der, password []byte) ([]byte, error) {
	var encrypted encryptedPKCS8PrivateKey
	if _, err := asn1.Unmarshal(der, &encrypted); err != nil {
		return nil, fmt.Errorf("decode encrypted PKCS#8: %w", err)
	}
	if !encrypted.EncryptionAlgorithm.Algorithm.Equal(asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 5, 13}) {
		return nil, fmt.Errorf("unsupported PKCS#8 encryption algorithm %s", encrypted.EncryptionAlgorithm.Algorithm.String())
	}

	var pbes2 pbes2Parameters
	if _, err := asn1.Unmarshal(encrypted.EncryptionAlgorithm.Parameters.FullBytes, &pbes2); err != nil {
		return nil, fmt.Errorf("decode PBES2 parameters: %w", err)
	}
	if !pbes2.KeyDerivationFunc.Algorithm.Equal(asn1.ObjectIdentifier{1, 2, 840, 113549, 1, 5, 12}) {
		return nil, fmt.Errorf("unsupported PBES2 key derivation algorithm %s", pbes2.KeyDerivationFunc.Algorithm.String())
	}
	if !pbes2.EncryptionScheme.Algorithm.Equal(asn1.ObjectIdentifier{2, 16, 840, 1, 101, 3, 4, 1, 42}) {
		return nil, fmt.Errorf("unsupported PBES2 encryption scheme %s", pbes2.EncryptionScheme.Algorithm.String())
	}

	var kdf pbkdf2Parameters
	if _, err := asn1.Unmarshal(pbes2.KeyDerivationFunc.Parameters.FullBytes, &kdf); err != nil {
		return nil, fmt.Errorf("decode PBKDF2 parameters: %w", err)
	}
	var salt []byte
	if _, err := asn1.Unmarshal(kdf.Salt.FullBytes, &salt); err != nil {
		return nil, fmt.Errorf("decode PBKDF2 salt: %w", err)
	}
	if kdf.IterationCount <= 0 {
		return nil, fmt.Errorf("invalid PBKDF2 iteration count %d", kdf.IterationCount)
	}
	keyLength := kdf.KeyLength
	if keyLength == 0 {
		keyLength = 32
	}
	key := pbkdf2.Key(password, salt, kdf.IterationCount, keyLength, sha256.New)

	var iv []byte
	if _, err := asn1.Unmarshal(pbes2.EncryptionScheme.Parameters.FullBytes, &iv); err != nil {
		return nil, fmt.Errorf("decode AES IV: %w", err)
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("create AES cipher: %w", err)
	}
	if len(iv) != block.BlockSize() || len(encrypted.EncryptedData) == 0 || len(encrypted.EncryptedData)%block.BlockSize() != 0 {
		return nil, errors.New("invalid AES-CBC encrypted PKCS#8 data")
	}
	plaintext := make([]byte, len(encrypted.EncryptedData))
	cipher.NewCBCDecrypter(block, iv).CryptBlocks(plaintext, encrypted.EncryptedData)
	padding := int(plaintext[len(plaintext)-1])
	if padding == 0 || padding > block.BlockSize() || padding > len(plaintext) {
		return nil, errors.New("invalid PKCS#7 padding in encrypted PKCS#8 data")
	}
	for _, value := range plaintext[len(plaintext)-padding:] {
		if int(value) != padding {
			return nil, errors.New("invalid PKCS#7 padding in encrypted PKCS#8 data")
		}
	}
	return plaintext[:len(plaintext)-padding], nil
}
