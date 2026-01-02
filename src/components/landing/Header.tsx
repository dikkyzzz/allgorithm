"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Github } from "lucide-react";

export const Header = () => {
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    return (
        <header
            className={cn(
                "fixed top-0 w-full z-50 transition-all duration-300 border-b border-transparent",
                scrolled
                    ? "bg-white/80 backdrop-blur-md border-gray-200 py-3 shadow-sm"
                    : "bg-transparent py-5"
            )}
        >
            <div className="container max-w-6xl mx-auto px-4 md:px-6 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-2 group">
                    <Image
                        src="/logo.png"
                        alt="ALLgorithm Logo"
                        width={48}
                        height={48}
                        className="rounded-lg"
                    />
                    <span className="font-bold text-xl tracking-tight text-gray-900">
                        ALL<span className="text-[#004040]">gorithm</span>
                    </span>
                </Link>

                <nav className="hidden md:flex items-center gap-8">
                    {[
                        { label: "Algorithms", href: "/#algorithms" },
                        { label: "Demo", href: "/#demo" },
                        { label: "Use Cases", href: "/#use-cases" },
                        { label: "Docs", href: "/#docs" },
                    ].map((item) => (
                        <Link
                            key={item.label}
                            href={item.href}
                            className="text-sm font-medium text-gray-600 hover:text-[#004040] transition-colors"
                        >
                            {item.label}
                        </Link>
                    ))}
                </nav>

                <div className="flex items-center gap-4">
                    <Link href="https://github.com" target="_blank">
                        <Button variant="ghost" className="text-gray-600 hover:text-[#004040]">
                            <Github className="w-5 h-5 mr-2" />
                            GitHub
                        </Button>
                    </Link>
                    <Link href="/#demo">
                        <Button className="bg-[#004040] hover:bg-[#003333] text-white border-0">
                            Try Demo
                        </Button>
                    </Link>
                </div>
            </div>
        </header>
    );
};
