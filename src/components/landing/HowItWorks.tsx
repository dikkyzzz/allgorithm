"use client";

import React from "react";
import { motion } from "framer-motion";
import { Upload, SlidersHorizontal, Play, FileText, ArrowRight } from "lucide-react";

const steps = [
    {
        icon: Upload,
        title: "Upload Your Data",
        description: "Import CSV files or choose from our sample datasets to get started quickly.",
        step: "01"
    },
    {
        icon: SlidersHorizontal,
        title: "Configure Parameters",
        description: "Fine-tune algorithm settings with intuitive sliders and dropdowns.",
        step: "02"
    },
    {
        icon: Play,
        title: "Run Analysis",
        description: "Execute algorithms directly in your browser with real-time progress.",
        step: "03"
    },
    {
        icon: FileText,
        title: "Interpret Results",
        description: "Get detailed metrics, visualizations, and plain-language explanations.",
        step: "04"
    }
];

export const HowItWorks = () => {
    return (
        <section className="py-24 md:py-32 bg-[#004040] relative overflow-hidden">
            {/* Background decoration */}
            <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-0 left-0 w-96 h-96 bg-white/5 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2" />
                <div className="absolute bottom-0 right-0 w-96 h-96 bg-white/5 rounded-full blur-3xl translate-x-1/2 translate-y-1/2" />
                {/* Grid pattern */}
                <div className="absolute inset-0 opacity-10" style={{
                    backgroundImage: 'radial-gradient(circle at 1px 1px, white 1px, transparent 0)',
                    backgroundSize: '40px 40px'
                }} />
            </div>

            <div className="container max-w-6xl mx-auto px-4 md:px-6 relative z-10">
                <motion.div
                    className="text-center mb-16"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                >
                    <span className="inline-block px-4 py-1 rounded-full bg-white/10 text-white/80 text-sm font-medium mb-4">
                        Simple Process
                    </span>
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">How It Works</h2>
                    <p className="text-white/70 text-lg max-w-2xl mx-auto">
                        From data to insights in four simple steps.
                    </p>
                </motion.div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
                    {/* Connector arrows - positioned between columns */}
                    {[0, 1, 2].map((i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, scaleX: 0 }}
                            whileInView={{ opacity: 1, scaleX: 1 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.4, delay: 0.6 + i * 0.15 }}
                            className="hidden md:flex absolute items-center z-0"
                            style={{
                                top: '56px', // Center of the 28x28 icon box
                                left: `calc(${(i + 1) * 25}% - 20px)`,
                                width: '40px'
                            }}
                        >
                            <div className="flex-1 h-0.5 bg-white/30" />
                            <ArrowRight className="w-4 h-4 text-white/50 flex-shrink-0" />
                        </motion.div>
                    ))}

                    {steps.map((step, idx) => {
                        const Icon = step.icon;
                        return (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: idx * 0.15 }}
                                whileHover={{ y: -8 }}
                                className="relative text-center group"
                            >

                                {/* Icon box */}
                                <div className="w-28 h-28 mx-auto mb-6 rounded-3xl bg-gradient-to-br from-white/20 to-white/5 border border-white/30 flex items-center justify-center text-white relative z-10 group-hover:from-white/30 group-hover:to-white/10 group-hover:border-white/50 transition-all duration-300 shadow-2xl">
                                    <Icon className="w-10 h-10 group-hover:scale-110 transition-transform duration-300" />
                                </div>

                                {/* Title */}
                                <h3 className="text-xl font-bold text-white mb-3 group-hover:text-white transition-colors">
                                    {step.title}
                                </h3>

                                {/* Description */}
                                <p className="text-white/60 text-sm leading-relaxed group-hover:text-white/80 transition-colors">
                                    {step.description}
                                </p>

                                {/* Glow effect */}
                                <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-32 h-32 bg-white/10 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                            </motion.div>
                        );
                    })}
                </div>

                {/* Bottom CTA hint */}
                <motion.div
                    className="text-center mt-16"
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.8 }}
                >
                    <p className="text-white/50 text-sm">
                        No installation required. Everything runs in your browser.
                    </p>
                </motion.div>
            </div>
        </section>
    );
};
