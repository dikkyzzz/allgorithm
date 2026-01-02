"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { algorithms } from "@/lib/algorithms";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowUpRight, ArrowRight, Sparkles } from "lucide-react";

// Featured algorithms to show on landing page (8 popular ones)
const featuredIds = [
    'k-means',
    'neural-network',
    'quick-sort',
    'binary-search',
    'dijkstra',
    'sha-256',
    'svm',
    'transformer'
];

const categoryColors: Record<string, string> = {
    'Clustering': 'bg-blue-100 text-blue-700',
    'Classification': 'bg-violet-100 text-violet-700',
    'Regression': 'bg-emerald-100 text-emerald-700',
    'Dimensionality Reduction': 'bg-amber-100 text-amber-700',
    'Text Mining': 'bg-rose-100 text-rose-700',
    'Sorting': 'bg-orange-100 text-orange-700',
    'Searching': 'bg-cyan-100 text-cyan-700',
    'Graph': 'bg-indigo-100 text-indigo-700',
    'Deep Learning': 'bg-purple-100 text-purple-700',
    'Statistics': 'bg-teal-100 text-teal-700',
    'Security': 'bg-red-100 text-red-700',
    'Recommendation': 'bg-pink-100 text-pink-700',
};

export const AlgorithmExplorer = () => {
    const featuredAlgorithms = algorithms.filter(a => featuredIds.includes(a.id));
    const totalCount = algorithms.length;
    const categoryCount = new Set(algorithms.map(a => a.category)).size;

    return (
        <section id="algorithms" className="py-24 md:py-32 bg-white">
            <div className="container max-w-6xl mx-auto px-4 md:px-6">
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 mb-12">
                    <div className="max-w-2xl">
                        <div className="flex items-center gap-2 mb-4">
                            <Sparkles className="w-5 h-5 text-[#004040]" />
                            <span className="text-sm font-medium text-[#004040]">Featured Algorithms</span>
                        </div>
                        <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-4">
                            Explore Our Library
                        </h2>
                        <p className="text-gray-600 text-lg">
                            From foundational statistical methods to advanced machine learning models,
                            find the right tool for your data discovery.
                        </p>
                    </div>

                    <div className="flex flex-col items-start md:items-end gap-2">
                        <div className="text-4xl font-bold text-[#004040]">{totalCount}</div>
                        <div className="text-sm text-gray-500">
                            Algorithms in {categoryCount} categories
                        </div>
                    </div>
                </div>

                {/* Featured Algorithms Grid */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12"
                >
                    {featuredAlgorithms.map((algo, index) => (
                        <motion.div
                            key={algo.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.05 }}
                        >
                            <Link href={`/algorithms/${algo.id}`}>
                                <Card className="h-full bg-white border-gray-200 hover:border-[#004040]/50 hover:shadow-lg transition-all group cursor-pointer">
                                    <CardHeader>
                                        <div className="flex justify-between items-start mb-2">
                                            <Badge className={categoryColors[algo.category] || 'bg-gray-100 text-gray-600'}>
                                                {algo.category}
                                            </Badge>
                                            <ArrowUpRight className="w-4 h-4 text-gray-400 group-hover:text-[#004040] transition-colors" />
                                        </div>
                                        <CardTitle className="text-xl text-gray-900 group-hover:text-[#004040] transition-colors">
                                            {algo.name}
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <CardDescription className="text-gray-600 line-clamp-2">
                                            {algo.description}
                                        </CardDescription>
                                    </CardContent>
                                    <CardFooter className="pt-0">
                                        <div className="text-xs text-gray-500 italic">
                                            {algo.bestFor}
                                        </div>
                                    </CardFooter>
                                </Card>
                            </Link>
                        </motion.div>
                    ))}
                </motion.div>

                {/* View All Button */}
                <div className="flex justify-center">
                    <Link href="/algorithms">
                        <Button
                            size="lg"
                            className="bg-[#004040] hover:bg-[#003030] text-white px-8 py-6 text-lg rounded-full shadow-lg hover:shadow-xl transition-all"
                        >
                            View All {totalCount} Algorithms
                            <ArrowRight className="w-5 h-5 ml-2" />
                        </Button>
                    </Link>
                </div>
            </div>
        </section>
    );
};
