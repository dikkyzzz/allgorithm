"use client";

import React from "react";
import { motion } from "framer-motion";

const stats = [
    { value: "20+", label: "Algorithms", desc: "From clustering to deep learning" },
    { value: "100K+", label: "Analyses Run", desc: "By students and researchers" },
    { value: "0ms", label: "Setup Time", desc: "Run directly in browser" },
    { value: "100%", label: "Open Source", desc: "Free forever, MIT license" },
];

export const Stats = () => {
    return (
        <section className="py-16 md:py-24 bg-gray-50 border-y border-gray-200">
            <div className="container max-w-6xl mx-auto px-4 md:px-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
                    {stats.map((stat, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: idx * 0.1 }}
                            className="text-center"
                        >
                            <div className="text-4xl md:text-5xl font-bold text-[#004040] mb-2">
                                {stat.value}
                            </div>
                            <div className="text-sm font-semibold text-gray-900 mb-1">{stat.label}</div>
                            <div className="text-xs text-gray-500">{stat.desc}</div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};
