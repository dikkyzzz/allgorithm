"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Github, Twitter, Linkedin } from "lucide-react";

export const Footer = () => {
    return (
        <footer className="bg-gray-50 pt-24 pb-12 border-t border-gray-200">
            <div className="container max-w-6xl mx-auto px-4 md:px-6">
                {/* CTA Banner */}
                <div className="bg-gradient-to-br from-[#004040] to-[#006666] rounded-3xl p-8 md:p-16 mb-24 relative overflow-hidden text-center md:text-left">
                    <div className="absolute top-0 left-0 w-full h-full bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 mix-blend-overlay" />
                    <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-8">
                        <div className="max-w-xl text-white">
                            <h2 className="text-3xl md:text-5xl font-bold mb-4">Start experimenting today.</h2>
                            <p className="text-white/80 text-lg">
                                Join thousands of researchers and students uncovering insights from data.
                            </p>
                        </div>
                        <div className="flex flex-col sm:flex-row gap-4">
                            <Link href="#demo">
                                <Button size="lg" className="bg-white text-[#004040] hover:bg-gray-100 h-14 px-8 text-base font-bold shadow-xl">
                                    Get Started for Free
                                </Button>
                            </Link>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16 px-4">
                    <div className="col-span-1 md:col-span-1">
                        <Link href="/" className="flex items-center gap-2 mb-6 group">
                            <Image
                                src="/logo.png"
                                alt="ALLgorithm Logo"
                                width={44}
                                height={44}
                                className="rounded-lg"
                            />
                            <span className="font-bold text-xl tracking-tight text-gray-900">
                                ALLgorithm
                            </span>
                        </Link>
                        <p className="text-gray-500 text-sm leading-relaxed mb-6">
                            The modern laboratory for data science education and rapid algorithm experimentation.
                        </p>
                        <div className="flex gap-4">
                            <Link href="#" className="text-gray-400 hover:text-[#004040] transition-colors">
                                <Github className="w-5 h-5" />
                            </Link>
                            <Link href="#" className="text-gray-400 hover:text-[#004040] transition-colors">
                                <Twitter className="w-5 h-5" />
                            </Link>
                            <Link href="#" className="text-gray-400 hover:text-[#004040] transition-colors">
                                <Linkedin className="w-5 h-5" />
                            </Link>
                        </div>
                    </div>

                    <div>
                        <h4 className="text-gray-900 font-bold mb-6">Product</h4>
                        <ul className="space-y-4">
                            {["Algorithms", "Interactive Demo", "Dataset Hub", "Export Utils"].map((item) => (
                                <li key={item}>
                                    <Link href="#" className="text-gray-500 hover:text-[#004040] text-sm transition-colors">
                                        {item}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div>
                        <h4 className="text-gray-900 font-bold mb-6">Resources</h4>
                        <ul className="space-y-4">
                            {["Documentation", "API Reference", "Methodology", "Privacy Policy"].map((item) => (
                                <li key={item}>
                                    <Link href="#" className="text-gray-500 hover:text-[#004040] text-sm transition-colors">
                                        {item}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div>
                        <h4 className="text-gray-900 font-bold mb-6">Open Source</h4>
                        <ul className="space-y-4">
                            {["GitHub Repository", "Contribute", "Report Issues", "MIT License"].map((item) => (
                                <li key={item}>
                                    <Link href="#" className="text-gray-500 hover:text-[#004040] text-sm transition-colors">
                                        {item}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                <div className="border-t border-gray-200 pt-8 flex flex-col md:flex-row justify-between items-center gap-4 text-center">
                    <p className="text-gray-500 text-xs">
                        © {new Date().getFullYear()} ALLgorithm. Built for the data science community.
                    </p>
                    <p className="text-gray-400 text-xs">
                        Powered by <span className="font-semibold text-[#004040]">Exmine-</span>
                    </p>
                    <div className="flex gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                        <span className="text-[10px] text-gray-500 font-mono tracking-widest uppercase">Open Source</span>
                    </div>
                </div>
            </div>
        </footer>
    );
};
