"use client";

import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Lock, Unlock, Key, Shield, Hash, RefreshCw } from 'lucide-react';

interface SecurityVisualizerProps {
    algorithmId: string;
}

export const SecurityVisualizer = ({ algorithmId }: SecurityVisualizerProps) => {
    const [inputText, setInputText] = useState('Hello, World!');
    const [outputText, setOutputText] = useState('');
    const [key, setKey] = useState('my-secret-key-123');
    const [isEncrypting, setIsEncrypting] = useState(false);
    const [step, setStep] = useState<'idle' | 'processing' | 'complete'>('idle');
    const [animationSteps, setAnimationSteps] = useState<string[]>([]);

    // DH specific
    const [dhParams, setDhParams] = useState<{ p: number, g: number, a: number, b: number, A: number, B: number, sharedA: number, sharedB: number } | null>(null);

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    // Simple XOR-based encryption simulation (for demo purposes)
    const simpleEncrypt = (text: string, key: string): string => {
        let result = '';
        for (let i = 0; i < text.length; i++) {
            const charCode = text.charCodeAt(i) ^ key.charCodeAt(i % key.length);
            result += charCode.toString(16).padStart(2, '0');
        }
        return result.toUpperCase();
    };

    // Simple hash simulation
    const simpleHash = (text: string): string => {
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            const char = text.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        // Convert to hex-like string
        const hex = Math.abs(hash).toString(16).padStart(8, '0');
        return (hex + hex + hex + hex + hex + hex + hex + hex).toUpperCase().substring(0, 64);
    };

    // RSA-like demo
    const rsaDemo = (text: string): { encrypted: string, publicKey: string, privateKey: string } => {
        const p = 61, q = 53;
        const n = p * q;
        const e = 17;

        // Simple encoding
        const encrypted = Array.from(text).map(c => {
            return Math.pow(c.charCodeAt(0), e) % n;
        }).join('-');

        return {
            encrypted,
            publicKey: `(n=${n}, e=${e})`,
            privateKey: `(p=${p}, q=${q})`
        };
    };

    // Diffie-Hellman demo
    const runDiffieHellman = async () => {
        setIsEncrypting(true);
        setStep('processing');
        setAnimationSteps([]);

        const p = 23; // Prime
        const g = 5;  // Generator
        const a = 6;  // Alice's private
        const b = 15; // Bob's private

        setAnimationSteps(prev => [...prev, `Public parameters: p=${p}, g=${g}`]);
        await sleep(500);

        const A = Math.pow(g, a) % p; // Alice's public
        setAnimationSteps(prev => [...prev, `Alice computes: A = g^a mod p = ${g}^${a} mod ${p} = ${A}`]);
        await sleep(500);

        const B = Math.pow(g, b) % p; // Bob's public
        setAnimationSteps(prev => [...prev, `Bob computes: B = g^b mod p = ${g}^${b} mod ${p} = ${B}`]);
        await sleep(500);

        const sharedA = Math.pow(B, a) % p; // Alice's shared secret
        setAnimationSteps(prev => [...prev, `Alice computes: K = B^a mod p = ${B}^${a} mod ${p} = ${sharedA}`]);
        await sleep(500);

        const sharedB = Math.pow(A, b) % p; // Bob's shared secret
        setAnimationSteps(prev => [...prev, `Bob computes: K = A^b mod p = ${A}^${b} mod ${p} = ${sharedB}`]);
        await sleep(500);

        setAnimationSteps(prev => [...prev, `✓ Shared secret established: K = ${sharedA}`]);

        setDhParams({ p, g, a, b, A, B, sharedA, sharedB });
        setStep('complete');
        setIsEncrypting(false);
    };

    const process = async () => {
        if (algorithmId === 'diffie-hellman') {
            runDiffieHellman();
            return;
        }

        setIsEncrypting(true);
        setStep('processing');
        setAnimationSteps([]);

        await sleep(300);
        setAnimationSteps(prev => [...prev, 'Reading input data...']);
        await sleep(400);

        let result = '';

        switch (algorithmId) {
            case 'aes':
                setAnimationSteps(prev => [...prev, 'Generating round keys...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Performing SubBytes...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Performing ShiftRows...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Performing MixColumns...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Adding round key...']);
                await sleep(300);
                result = simpleEncrypt(inputText, key);
                setAnimationSteps(prev => [...prev, '✓ Encryption complete!']);
                break;

            case 'rsa':
                setAnimationSteps(prev => [...prev, 'Generating prime numbers...']);
                await sleep(400);
                setAnimationSteps(prev => [...prev, 'Computing n = p × q...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Computing public exponent e...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Encrypting with public key...']);
                await sleep(400);
                const rsaResult = rsaDemo(inputText);
                result = rsaResult.encrypted;
                setAnimationSteps(prev => [...prev, `Public Key: ${rsaResult.publicKey}`]);
                setAnimationSteps(prev => [...prev, '✓ RSA encryption complete!']);
                break;

            case 'sha-256':
                setAnimationSteps(prev => [...prev, 'Padding message...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Initializing hash values...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Processing 64 rounds...']);
                await sleep(500);
                setAnimationSteps(prev => [...prev, 'Computing final hash...']);
                await sleep(300);
                result = simpleHash(inputText);
                setAnimationSteps(prev => [...prev, '✓ Hash computed!']);
                break;

            case 'hmac':
                setAnimationSteps(prev => [...prev, 'Preparing key (padding/hashing)...']);
                await sleep(300);
                setAnimationSteps(prev => [...prev, 'Computing inner hash: H(K ⊕ ipad || message)...']);
                await sleep(400);
                setAnimationSteps(prev => [...prev, 'Computing outer hash: H(K ⊕ opad || inner)...']);
                await sleep(400);
                result = simpleHash(key + inputText);
                setAnimationSteps(prev => [...prev, '✓ HMAC computed!']);
                break;

            default:
                result = simpleEncrypt(inputText, key);
        }

        setOutputText(result);
        setStep('complete');
        setIsEncrypting(false);
    };

    const algorithmName = {
        'aes': 'AES Encryption',
        'rsa': 'RSA Encryption',
        'sha-256': 'SHA-256 Hash',
        'hmac': 'HMAC',
        'diffie-hellman': 'Diffie-Hellman Key Exchange'
    }[algorithmId] || 'Encryption';

    const isHash = ['sha-256', 'hmac'].includes(algorithmId);
    const isDH = algorithmId === 'diffie-hellman';

    return (
        <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-lg font-bold text-gray-900">
                    <Shield className="w-5 h-5 inline mr-2 text-red-600" />
                    {algorithmName}
                </h3>
                <Button
                    size="sm"
                    onClick={process}
                    disabled={isEncrypting}
                    className="bg-[#004040] hover:bg-[#003030]"
                >
                    {isDH ? <Key className="w-4 h-4 mr-1" /> : isHash ? <Hash className="w-4 h-4 mr-1" /> : <Lock className="w-4 h-4 mr-1" />}
                    {isEncrypting ? 'Processing...' : isDH ? 'Exchange Keys' : isHash ? 'Compute Hash' : 'Encrypt'}
                </Button>
            </div>

            {/* Input Fields */}
            {!isDH && (
                <div className="space-y-4">
                    <div className="space-y-2">
                        <label className="text-sm text-gray-600">Input Text</label>
                        <Input
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            placeholder="Enter text to encrypt/hash..."
                        />
                    </div>
                    {!isHash && algorithmId !== 'rsa' && (
                        <div className="space-y-2">
                            <label className="text-sm text-gray-600">Secret Key</label>
                            <Input
                                value={key}
                                onChange={(e) => setKey(e.target.value)}
                                placeholder="Enter secret key..."
                                type="password"
                            />
                        </div>
                    )}
                </div>
            )}

            {/* DH Explanation */}
            {isDH && (
                <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
                    <p className="text-sm text-amber-700">
                        <strong>Diffie-Hellman</strong> allows two parties (Alice & Bob) to establish a shared secret over an insecure channel without ever transmitting the secret itself.
                    </p>
                </div>
            )}

            {/* Animation Steps */}
            {animationSteps.length > 0 && (
                <div className="p-4 bg-gray-900 rounded-lg font-mono text-sm space-y-1 max-h-48 overflow-y-auto">
                    {animationSteps.map((step, i) => (
                        <div key={i} className={`${step.startsWith('✓') ? 'text-emerald-400' : 'text-gray-300'}`}>
                            {step}
                        </div>
                    ))}
                </div>
            )}

            {/* Output */}
            {step === 'complete' && !isDH && (
                <div className="space-y-2">
                    <label className="text-sm text-gray-600">{isHash ? 'Hash Output' : 'Encrypted Output'}</label>
                    <div className="p-4 bg-gray-100 rounded-lg font-mono text-sm break-all">
                        {outputText}
                    </div>
                </div>
            )}

            {/* DH Results */}
            {step === 'complete' && isDH && dhParams && (
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <div className="text-xs text-blue-600 font-bold uppercase mb-2">Alice</div>
                        <div className="space-y-1 text-sm">
                            <div>Private (a): <span className="font-mono">{dhParams.a}</span></div>
                            <div>Public (A): <span className="font-mono">{dhParams.A}</span></div>
                            <div>Shared (K): <span className="font-mono font-bold text-emerald-600">{dhParams.sharedA}</span></div>
                        </div>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                        <div className="text-xs text-purple-600 font-bold uppercase mb-2">Bob</div>
                        <div className="space-y-1 text-sm">
                            <div>Private (b): <span className="font-mono">{dhParams.b}</span></div>
                            <div>Public (B): <span className="font-mono">{dhParams.B}</span></div>
                            <div>Shared (K): <span className="font-mono font-bold text-emerald-600">{dhParams.sharedB}</span></div>
                        </div>
                    </div>
                </div>
            )}

            {/* Visual representation */}
            {step === 'complete' && (
                <div className="flex justify-center items-center gap-4 py-4">
                    <div className="text-center">
                        <div className="w-16 h-16 bg-blue-100 rounded-lg flex items-center justify-center mb-2">
                            <span className="text-2xl">📄</span>
                        </div>
                        <span className="text-xs text-gray-500">{isDH ? 'Alice' : 'Plaintext'}</span>
                    </div>
                    <div className="flex items-center">
                        <div className="w-12 h-0.5 bg-gray-300"></div>
                        <div className="px-3 py-1 bg-red-100 rounded-full">
                            {isDH ? <Key className="w-4 h-4 text-red-600" /> : isHash ? <Hash className="w-4 h-4 text-red-600" /> : <Lock className="w-4 h-4 text-red-600" />}
                        </div>
                        <div className="w-12 h-0.5 bg-gray-300"></div>
                    </div>
                    <div className="text-center">
                        <div className="w-16 h-16 bg-emerald-100 rounded-lg flex items-center justify-center mb-2">
                            <span className="text-2xl">{isDH ? '🤝' : '🔐'}</span>
                        </div>
                        <span className="text-xs text-gray-500">{isDH ? 'Shared Key' : isHash ? 'Hash' : 'Ciphertext'}</span>
                    </div>
                </div>
            )}
        </div>
    );
};
