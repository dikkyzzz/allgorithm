"use client";

import React from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

const faqs = [
    {
        q: "Do I need to install Python or R to run these?",
        a: "No. All algorithms run directly in your browser using optimized JavaScript. No setup required."
    },
    {
        q: "Is my data secure?",
        a: "Absolutely. All processing happens locally in your browser. We never upload or store your data on any server."
    },
    {
        q: "Which algorithms are currently supported?",
        a: "We support over 20+ algorithms including K-Means, Naive Bayes, PCA, SVM, and various Regression models. We are constantly adding more."
    },
    {
        q: "Can I export the results of my analysis?",
        a: "Yes. You can export detailed reports in JSON format for use in other tools."
    },
    {
        q: "Does it support multi-class classification?",
        a: "Yes, our Naive Bayes and SVM implementations fully support multi-class classification out of the box."
    },
    {
        q: "Is there a limit to the dataset size?",
        a: "You can process datasets up to 10MB directly in the browser. For larger datasets, consider sampling."
    }
];

export const FAQ = () => {
    return (
        <section id="docs" className="py-24 md:py-32 bg-gray-50">
            <div className="container max-w-4xl mx-auto px-4 md:px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-5xl font-bold text-gray-900 mb-6">Frequently Asked</h2>
                    <p className="text-gray-600 text-lg">
                        Everything you need to know about ALLgorithm.
                    </p>
                </div>

                <Accordion type="single" collapsible className="w-full space-y-4">
                    {faqs.map((faq, idx) => (
                        <AccordionItem
                            key={idx}
                            value={`item-${idx}`}
                            className="border border-gray-200 bg-white px-6 rounded-2xl overflow-hidden data-[state=open]:border-[#004040]/30 data-[state=open]:shadow-md transition-all"
                        >
                            <AccordionTrigger className="text-gray-900 hover:text-[#004040] hover:no-underline font-medium text-left">
                                {faq.q}
                            </AccordionTrigger>
                            <AccordionContent className="text-gray-600 leading-relaxed pb-6">
                                {faq.a}
                            </AccordionContent>
                        </AccordionItem>
                    ))}
                </Accordion>
            </div>
        </section>
    );
};
