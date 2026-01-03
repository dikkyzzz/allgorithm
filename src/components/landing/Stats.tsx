"use client";

import React from "react";
import { motion } from "framer-motion";
import { Brain, Zap, Clock, Heart } from "lucide-react";

const stats = [
    {
        value: "20+",
        label: "Algorithms",
        desc: "From clustering to deep learning",
        icon: Brain
    },
    {
        value: "100K+",
        label: "Analyses Run",
        desc: "By students and researchers",
        icon: Zap
    },
    {
        value: "0ms",
        label: "Setup Time",
        desc: "Run directly in browser",
        icon: Clock
    },
    {
        value: "100%",
        label: "Open Source",
        desc: "Free forever, MIT license",
        icon: Heart
    },
];

export const Stats = () => {
    return (
        <section className="py-16 md:py-24 bg-gradient-to-b from-gray-50 to-white border-y border-gray-200 relative overflow-hidden">
            {/* Background Pattern */}
            <div className="absolute inset-0 opacity-20">
                <div className="absolute top-0 left-1/4 w-72 h-72 bg-[#004040] rounded-full blur-3xl" />
                <div className="absolute bottom-0 right-1/4 w-72 h-72 bg-[#006060] rounded-full blur-3xl" />
            </div>

            <div className="container max-w-6xl mx-auto px-4 md:px-6 relative z-10">
                {/* Section Title */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-12"
                >
                    <h2 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">
                        Trusted by Learners Worldwide
                    </h2>
                    <p className="text-gray-500">Join thousands exploring algorithms interactively</p>
                </motion.div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
                    {stats.map((stat, idx) => {
                        const Icon = stat.icon;
                        return (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 30, scale: 0.9 }}
                                whileInView={{ opacity: 1, y: 0, scale: 1 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: idx * 0.1 }}
                                whileHover={{ y: -5, scale: 1.02 }}
                                className="relative p-6 rounded-2xl bg-white border-2 border-[#004040]/20 shadow-lg hover:shadow-xl hover:border-[#004040]/40 transition-all duration-300 cursor-default group"
                            >
                                {/* Icon */}
                                <div className="w-12 h-12 rounded-xl mb-4 mx-auto bg-[#004040] flex items-center justify-center shadow-lg group-hover:scale-110 group-hover:bg-[#006060] transition-all duration-300">
                                    <Icon className="w-6 h-6 text-white" />
                                </div>

                                {/* Value */}
                                <div className="text-3xl md:text-4xl font-extrabold mb-1 text-[#004040] text-center">
                                    {stat.value}
                                </div>

                                {/* Label */}
                                <div className="text-sm font-bold text-gray-800 mb-1 text-center">
                                    {stat.label}
                                </div>

                                {/* Description */}
                                <div className="text-xs text-gray-500 text-center">
                                    {stat.desc}
                                </div>

                                {/* Glow effect on hover */}
                                <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-10 bg-[#004040] transition-opacity duration-300 pointer-events-none" />
                            </motion.div>
                        );
                    })}
                </div>
            </div>
        </section>
    );
};
