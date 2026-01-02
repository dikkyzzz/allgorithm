"use client";

import React from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Users, MailWarning, TrendingUp, SearchCode, ShoppingBag } from "lucide-react";

const useCases = [
    {
        icon: <Users className="w-6 h-6" />,
        title: "Customer Segmentation",
        desc: "Group customers based on purchasing behavior for targeted marketing.",
        algos: ["K-Means", "DBSCAN"],
        color: "text-blue-600"
    },
    {
        icon: <MailWarning className="w-6 h-6" />,
        title: "Spam Detection",
        desc: "Automatically filter out unwanted communications with high accuracy.",
        algos: ["Naive Bayes", "SVM"],
        color: "text-rose-600"
    },
    {
        icon: <TrendingUp className="w-6 h-6" />,
        title: "Sales Forecasting",
        desc: "Predict future revenue based on historical trends and variables.",
        algos: ["Linear Regression", "Ridge"],
        color: "text-emerald-600"
    },
    {
        icon: <SearchCode className="w-6 h-6" />,
        title: "Topic Discovery",
        desc: "Extract underlying themes from large volumes of unstructured text.",
        algos: ["TF-IDF + NB", "PCA"],
        color: "text-violet-600"
    },
    {
        icon: <ShoppingBag className="w-6 h-6" />,
        title: "Churn Prediction",
        desc: "Identify users likely to unsubscribe before they leave.",
        algos: ["Logistic Regression"],
        color: "text-amber-600"
    }
];

export const UseCases = () => {
    return (
        <section id="use-cases" className="py-24 md:py-32 bg-white">
            <div className="container max-w-6xl mx-auto px-4 md:px-6">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-16">
                    <div className="max-w-xl">
                        <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-6">Real-World Utility</h2>
                        <p className="text-gray-600 text-lg">
                            Scientific algorithms applied to practical business problems.
                            See how ALLgorithm can transform your workflow.
                        </p>
                    </div>
                    <Badge variant="outline" className="border-gray-300 text-gray-500 px-4 py-1">
                        50+ Industry Templates
                    </Badge>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {useCases.map((useCase, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, scale: 0.95 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: idx * 0.1 }}
                        >
                            <Card className="bg-white border-gray-200 h-full hover:shadow-lg hover:border-[#004040]/30 transition-all">
                                <CardContent className="p-8">
                                    <div className={`${useCase.color} mb-6`}>
                                        {useCase.icon}
                                    </div>
                                    <h3 className="text-xl font-bold text-gray-900 mb-3">{useCase.title}</h3>
                                    <p className="text-gray-600 text-sm mb-6 leading-relaxed">
                                        {useCase.desc}
                                    </p>
                                    <div className="space-y-2">
                                        <div className="text-[10px] uppercase tracking-widest font-bold text-gray-400">Recommended</div>
                                        <div className="flex flex-wrap gap-2">
                                            {useCase.algos.map((algo) => (
                                                <span key={algo} className="text-xs text-[#004040] font-medium">#{algo}</span>
                                            ))}
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};
