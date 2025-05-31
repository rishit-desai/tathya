/* eslint-disable @typescript-eslint/no-unused-vars */
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { motion } from 'framer-motion';
import { Send, BookOpen, BrainCircuit } from 'lucide-react';
import { cn } from "@/lib/utils";
import { ScrollArea } from './components/ui/scroll-area';
import ReactMarkdown from 'react-markdown';

// Animation Variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      delayChildren: 0.3,
      staggerChildren: 0.2
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1
  }
};

// Main App Component
const NLPApp = () => {
  const [inputParagraph, setInputParagraph] = useState<string>('');
  const [outputParagraph, setOutputParagraph] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  // Scroll to output when it updates
  useEffect(() => {
    if (outputParagraph && outputRef.current) {
      outputRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [outputParagraph]);

  // Simulate API call (replace with actual fetch in a real app)
  const processParagraph = useCallback(async () => {
    if (!inputParagraph.trim()) {
      setError("Please enter a paragraph to process.");
      return;
    }

    setLoading(true);
    setError(null);
    setOutputParagraph(''); // Clear previous output

    // Simulate a network request delay (replace with your actual API endpoint)
    try {
      // Replace 'http://your-flask-api/process_text' with your actual Flask API endpoint.
      const response = await fetch('http://localhost:5000/process_text', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({ paragraph: inputParagraph }),
      });

      if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setOutputParagraph(data.result);

      // Simulate a delay and a response.  This is ONLY for demonstration.
      // await new Promise(resolve => setTimeout(resolve, 2000));
      // const simulatedResponse: string = `Processed version of: ${inputParagraph}.  This is a simulated result from the NLP processor.  The input text has been analyzed and transformed.`;
      // setOutputParagraph(simulatedResponse);

    } catch (err: any) {
      setError(err.message || "An error occurred while processing your request.");
    } finally {
      setLoading(false);
    }
  }, [inputParagraph]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black p-4 sm:p-8 flex items-center justify-center">
      <div className="space-y-8 w-full max-w-4xl">
        <motion.div
          variants={itemVariants}
          initial="hidden"
          animate="visible"
          className="text-center space-y-4"
        >
          <h1
            className="text-4xl sm:text-5xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500"
          >
            <BrainCircuit className="inline-block w-8 h-8 sm:w-10 sm:h-10 md:w-12 md:h-12 mr-2" />
            Tathya<br />Explainable Bias Detection
          </h1>
          <p className="text-gray-400 text-base sm:text-lg md:text-xl">
            Enter an article or paragraph to analyze and detect bias.
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="space-y-6"
        >
          <motion.div variants={itemVariants}>
            <Textarea
              placeholder="Enter your paragraph here..."
              value={inputParagraph}
              onChange={(e) => setInputParagraph(e.target.value)}
              className={cn(
                "w-full min-h-[150px] sm:min-h-[175px] md:min-h-[200px] text-gray-300 bg-black/50",
                "border-gray-700 focus:border-blue-500 focus:ring-blue-500",
                "placeholder:text-gray-500",
                "shadow-lg"
              )}
              disabled={loading}
            />
          </motion.div>

          <motion.div variants={itemVariants} className="flex justify-center">
            <Button
              onClick={processParagraph}
              disabled={loading}
              className={cn(
                "bg-gradient-to-r from-blue-500 to-purple-500 text-white px-6 py-3 sm:px-7 sm:py-3.5 md:px-8 md:py-4",
                "rounded-full shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-300",
                "font-semibold text-base sm:text-lg md:text-xl",
                loading && "opacity-70 cursor-not-allowed"
              )}
            >
              {loading ? (
                <>
                  <svg
                    className="animate-spin h-5 w-5 mr-3 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Processing...
                </>
              ) : (
                <>
                  <Send className="mr-2 w-5 h-5" />
                  Process Paragraph
                </>
              )}
            </Button>
          </motion.div>
        </motion.div>

        {error && (
          <motion.div
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-lg shadow-md"
          >
            {error}
          </motion.div>
        )}

        {outputParagraph && (
          <motion.div
            variants={itemVariants}
            initial="hidden"
            animate="visible"
            className="bg-gray-800/50 border border-gray-700 p-4 sm:p-5 md:p-6 rounded-lg shadow-lg"
          >
            <h2 className="text-lg sm:text-xl font-semibold text-gray-200 mb-2 flex items-center">
              <BookOpen className="mr-2 w-5 h-5" />
              Processed Output:
            </h2>
              <div ref={outputRef} className="p-4">
                <ReactMarkdown
                  children={outputParagraph}
                  components={{
                    // Customize heading levels
                    h1: ({ node, ...props }) => <h1 {...props} className="text-2xl font-bold text-gray-50" />,
                    h2: ({ node, ...props }) => <h2 {...props} className="text-xl font-semibold text-gray-100" />,
                    h3: ({ node, ...props }) => <h3 {...props} className="text-lg font-semibold text-gray-200" />,
                    h4: ({ node, ...props }) => <h4 {...props} className="text-md font-semibold text-gray-300" />,
                    h5: ({ node, ...props }) => <h5 {...props} className="text-sm font-semibold text-gray-300" />,
                    h6: ({ node, ...props }) => <h6 {...props} className="text-xs font-semibold text-gray-300" />,

                    p: ({ node, ...props }) => <p {...props} className="text-gray-300" />,
                    a: ({ node, ...props }) => <a {...props} className="text-blue-400 hover:text-blue-300" />,
                    ul: ({ node, ...props }) => <ul {...props} className="list-disc list-inside text-gray-300" />,
                    ol: ({ node, ...props }) => <ol {...props} className="list-decimal list-inside text-gray-300" />,
                    li: ({ node, ...props }) => <li {...props} className="text-gray-300" />,
                    strong: ({ node, ...props }) => <strong {...props} className="text-white font-bold" />,
                    em: ({ node, ...props }) => <em {...props} className="text-gray-300 italic" />,
                    blockquote: ({ node, ...props }) => (
                      <blockquote
                        {...props}
                        className="border-l-4 border-gray-600 pl-4 text-gray-400 italic"
                      />
                    ),
                    code: ({ node, className, ...props }) => {
                      const codeProps = {
                        ...props,
                        className: cn(
                          "rounded-md px-[0.3rem] py-[0.2rem] font-mono text-sm",
                          "bg-gray-800 p-4 block rounded-md overflow-x-auto whitespace-pre-wrap",
                          className
                        )
                      };
                      return  <pre {...codeProps} />;
                    },
                    hr: ({ node, ...props }) => <hr {...props} className="border-t border-gray-700 my-4" />,
                    img: ({ node, ...props }) => <img {...props} className="rounded-md max-w-full h-auto mb-4" />,
                  }}
                />
              </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default NLPApp;
