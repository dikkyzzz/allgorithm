"use client";

import React from "react";
import { motion } from "framer-motion";
import { Upload, SlidersHorizontal, Play, FileText } from "lucide-react";

const steps = [
    {
        icon: <Upload className="w-6 h-6" />,
        title: "Upload Your Data",
        description: "Import CSV files or choose from our sample datasets to get started quickly."
    },
    {
        icon: <SlidersHorizontal className="w-6 h-6" />,
        title: "Configure Parameters",
        description: "Fine-tune algorithm settings with intuitive sliders and dropdowns."
    },
    {
        icon: <Play className="w-6 h-6" />,
        title: "Run Analysis",
        description: "Execute algorithms directly in your browser with real-time progress."
    },
    {
        icon: <FileText className="w-6 h-6" />,
        title: "Interpret Results",
        description: "Get detailed metrics, visualizations, and plain-language explanations."
    }
];

export const HowItWorks = () => {
    return (
        <section className="py-24 md:py-32 bg-[#004040]">
            <div className="container max-w-6xl mx-auto px-4 md:px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">How It Works</h2>
                    <p className="text-white/70 text-lg max-w-2xl mx-auto">
                        From data to insights in four simple steps.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative">
                    {/* Connector line */}
                    <div className="hidden md:block absolute top-12 left-[12.5%] right-[12.5%] h-0.5 bg-white/20" />

                    {steps.map((step, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: idx * 0.15 }}
                            className="relative text-center"
                        >
                            <div className="w-24 h-24 mx-auto mb-6 rounded-2xl bg-white/10 border border-white/20 flex items-center justify-center text-white relative z-10">
                                {step.icon}
                            </div>
                            <h3 className="text-xl font-bold text-white mb-3">{step.title}</h3>
                            <p className="text-white/60 text-sm leading-relaxed">{step.description}</p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};
