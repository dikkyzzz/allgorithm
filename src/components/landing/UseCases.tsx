"use client";

import React from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Users, MailWarning, TrendingUp, SearchCode, ShoppingBag, ArrowRight } from "lucide-react";

const useCases = [
    {
        icon: Users,
        title: "Customer Segmentation",
        desc: "Group customers based on purchasing behavior for targeted marketing.",
        algos: ["K-Means", "DBSCAN"],
    },
    {
        icon: MailWarning,
        title: "Spam Detection",
        desc: "Automatically filter out unwanted communications with high accuracy.",
        algos: ["Naive Bayes", "SVM"],
    },
    {
        icon: TrendingUp,
        title: "Sales Forecasting",
        desc: "Predict future revenue based on historical trends and variables.",
        algos: ["Linear Regression", "Ridge"],
    },
    {
        icon: SearchCode,
        title: "Topic Discovery",
        desc: "Extract underlying themes from large volumes of unstructured text.",
        algos: ["TF-IDF + NB", "PCA"],
    },
    {
        icon: ShoppingBag,
        title: "Churn Prediction",
        desc: "Identify users likely to unsubscribe before they leave.",
        algos: ["Logistic Regression"],
    }
];

export const UseCases = () => {
    return (
        <section id="use-cases" className="py-24 md:py-32 bg-gradient-to-b from-white to-gray-50 relative overflow-hidden">
            {/* Background decoration */}
            <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-1/4 -left-32 w-64 h-64 bg-[#004040]/5 rounded-full blur-3xl" />
                <div className="absolute bottom-1/4 -right-32 w-64 h-64 bg-[#004040]/5 rounded-full blur-3xl" />
            </div>

            <div className="container max-w-6xl mx-auto px-4 md:px-6 relative z-10">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-16">
                    <motion.div
                        className="max-w-xl"
                        initial={{ opacity: 0, x: -20 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                    >
                        <Badge className="bg-[#004040]/10 text-[#004040] border-0 mb-4">
                            Use Cases
                        </Badge>
                        <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-4">
                            Real-World <span className="text-[#004040]">Utility</span>
                        </h2>
                        <p className="text-gray-600 text-lg">
                            Scientific algorithms applied to practical business problems.
                            See how ALLgorithm can transform your workflow.
                        </p>
                    </motion.div>
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                    >
                        <Badge variant="outline" className="border-[#004040]/30 text-[#004040] px-4 py-2 text-sm font-semibold">
                            50+ Industry Templates
                        </Badge>
                    </motion.div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {useCases.map((useCase, idx) => {
                        const Icon = useCase.icon;
                        return (
                            <motion.div
                                key={idx}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: idx * 0.1 }}
                                whileHover={{ y: -8 }}
                            >
                                <Card className="bg-white border-2 border-gray-100 h-full hover:border-[#004040]/30 hover:shadow-xl transition-all duration-300 group overflow-hidden">
                                    <CardContent className="p-8 relative">
                                        {/* Hover gradient overlay */}
                                        <div className="absolute inset-0 bg-gradient-to-br from-[#004040]/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

                                        {/* Icon */}
                                        <div className="w-14 h-14 rounded-2xl bg-[#004040] flex items-center justify-center mb-6 group-hover:scale-110 group-hover:rotate-3 transition-transform duration-300 shadow-lg">
                                            <Icon className="w-7 h-7 text-white" />
                                        </div>

                                        {/* Title */}
                                        <h3 className="text-xl font-bold text-gray-900 mb-3 group-hover:text-[#004040] transition-colors">
                                            {useCase.title}
                                        </h3>

                                        {/* Description */}
                                        <p className="text-gray-600 text-sm mb-6 leading-relaxed">
                                            {useCase.desc}
                                        </p>

                                        {/* Recommended Algorithms */}
                                        <div className="space-y-3">
                                            <div className="text-[10px] uppercase tracking-widest font-bold text-[#004040]/60">
                                                Recommended
                                            </div>
                                            <div className="flex flex-wrap gap-2">
                                                {useCase.algos.map((algo) => (
                                                    <span
                                                        key={algo}
                                                        className="px-3 py-1 bg-[#004040]/10 rounded-full text-xs text-[#004040] font-semibold hover:bg-[#004040] hover:text-white transition-colors cursor-pointer"
                                                    >
                                                        {algo}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Arrow indicator */}
                                        <div className="absolute bottom-6 right-6 opacity-0 group-hover:opacity-100 transition-all duration-300 transform translate-x-2 group-hover:translate-x-0">
                                            <ArrowRight className="w-5 h-5 text-[#004040]" />
                                        </div>
                                    </CardContent>
                                </Card>
                            </motion.div>
                        );
                    })}
                </div>
            </div>
        </section>
    );
};
