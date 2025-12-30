'use client';

import { useRef } from 'react';
import { Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface FileUploadProps {
  onFileSelect: (blob: Blob) => void;
  accept?: string;
  className?: string;
  children?: React.ReactNode;
}

export function FileUpload({
  onFileSelect,
  accept = 'image/*',
  className,
  children,
}: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
      // Reset input so the same file can be selected again
      e.target.value = '';
    }
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        className="hidden"
      />
      {children ? (
        <div onClick={handleClick} className={cn('cursor-pointer', className)}>
          {children}
        </div>
      ) : (
        <Button variant="outline" onClick={handleClick} className={cn('gap-2', className)}>
          <Upload className="h-4 w-4" />
          Upload Image
        </Button>
      )}
    </>
  );
}
