'use client';

import React, { useEffect, useRef, useState } from 'react';

const WORDS = ['AI Tool', '3D Decoder', 'Paper AI', 'Neural Stack', 'Math Engine'];
const TYPE_MS = 85;
const DELETE_MS = 40;
const HOLD_MS = 1800;

/**
 * Terminal-style word cycler: types a word, holds, deletes, moves on.
 * The block cursor lives outside this component so it never unmounts.
 */
export const Typewriter: React.FC<{ className?: string }> = ({ className = '' }) => {
  const [text, setText] = useState('');
  const [wordIndex, setWordIndex] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  useEffect(() => {
    if (reduced) {
      setText(WORDS[0]);
      return;
    }
    const word = WORDS[wordIndex % WORDS.length];
    let t: ReturnType<typeof setTimeout>;

    if (!deleting && text === word) {
      t = setTimeout(() => setDeleting(true), HOLD_MS);
    } else if (deleting && text === '') {
      setDeleting(false);
      setWordIndex((i) => (i + 1) % WORDS.length);
      return;
    } else {
      const next = deleting ? word.slice(0, text.length - 1) : word.slice(0, text.length + 1);
      t = setTimeout(() => setText(next), deleting ? DELETE_MS : TYPE_MS);
    }
    return () => clearTimeout(t);
  }, [text, deleting, wordIndex, reduced]);

  return (
    <span className={className} aria-label="decoded, explained, simplified">
      <span aria-hidden="true">{text}</span>
    </span>
  );
};

/**
 * Types a single line of text once it scrolls into view.
 * Used for the terminal strip. Reduced motion prints instantly.
 */
export const TypeOnce: React.FC<{ text: string; speed?: number; className?: string }> = ({
  text,
  speed = 38,
  className = '',
}) => {
  const ref = useRef<HTMLSpanElement | null>(null);
  const [out, setOut] = useState('');
  const [started, setStarted] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setStarted(true);
          observer.disconnect();
        }
      },
      { threshold: 0.4 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!started) return;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      setOut(text);
      return;
    }
    let i = 0;
    const id = setInterval(() => {
      i += 1;
      setOut(text.slice(0, i));
      if (i >= text.length) clearInterval(id);
    }, speed);
    return () => clearInterval(id);
  }, [started, text, speed]);

  return (
    <span ref={ref} className={className}>
      {out}
      {out.length < text.length && <span className="cursor-blink ml-0.5 inline-block h-[0.9em] w-[0.55em] translate-y-[0.12em] bg-volt" />}
    </span>
  );
};
