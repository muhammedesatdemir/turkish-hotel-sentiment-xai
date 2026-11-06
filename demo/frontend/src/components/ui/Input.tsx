import { forwardRef, type InputHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...rest }, ref) {
    return (
      <input
        ref={ref}
        className={cn(
          "block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm",
          "placeholder:text-gray-400 focus:border-primary focus:outline-none",
          "focus:ring-2 focus:ring-primary-500/20 disabled:bg-gray-50 disabled:text-gray-500",
          className,
        )}
        {...rest}
      />
    );
  },
);

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  function Textarea({ className, ...rest }, ref) {
    return (
      <textarea
        ref={ref}
        className={cn(
          "block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm",
          "placeholder:text-gray-400 focus:border-primary focus:outline-none",
          "focus:ring-2 focus:ring-primary-500/20 disabled:bg-gray-50 disabled:text-gray-500",
          "resize-y",
          className,
        )}
        {...rest}
      />
    );
  },
);
