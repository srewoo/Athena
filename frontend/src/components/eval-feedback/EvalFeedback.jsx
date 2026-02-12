import { useState } from 'react';
import axios from 'axios';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';
import { toast } from 'sonner';
import { Star, Loader2, ThumbsUp, MessageSquare } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

export default function EvalFeedback({ evalPrompt, userId, onFeedbackSubmitted }) {
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [comment, setComment] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async () => {
    if (rating === 0) {
      toast.error('Please select a rating');
      return;
    }

    setSubmitting(true);
    try {
      await axios.post(`${API}/eval-feedback`, {
        eval_prompt_id: evalPrompt.id,
        rating: rating,
        comment: comment.trim() || null,
        user_id: userId || null
      });

      setSubmitted(true);
      toast.success('Feedback submitted', {
        description: 'Thank you for helping us improve!'
      });

      if (onFeedbackSubmitted) {
        onFeedbackSubmitted(rating, comment);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      toast.error('Failed to submit feedback', {
        description: error.response?.data?.detail || 'Please try again'
      });
    } finally {
      setSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="text-center py-4 space-y-2">
        <ThumbsUp className="w-8 h-8 mx-auto text-green-500" />
        <p className="text-sm font-medium text-green-600">Feedback Submitted!</p>
        <p className="text-xs text-muted-foreground">
          Your rating helps improve future eval generation
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4 py-2">
      <div>
        <p className="text-sm font-medium mb-2">Rate this evaluation prompt</p>
        <div className="flex items-center space-x-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              type="button"
              onClick={() => setRating(star)}
              onMouseEnter={() => setHoverRating(star)}
              onMouseLeave={() => setHoverRating(0)}
              className="focus:outline-none transition-transform hover:scale-110"
            >
              <Star
                className={`w-8 h-8 ${
                  star <= (hoverRating || rating)
                    ? 'fill-yellow-400 text-yellow-400'
                    : 'text-gray-300'
                } transition-colors`}
              />
            </button>
          ))}
          {rating > 0 && (
            <span className="ml-2 text-sm font-medium">
              {rating === 1 && 'Poor'}
              {rating === 2 && 'Fair'}
              {rating === 3 && 'Good'}
              {rating === 4 && 'Very Good'}
              {rating === 5 && 'Excellent'}
            </span>
          )}
        </div>
      </div>

      <div>
        <div className="flex items-center space-x-2 mb-2">
          <MessageSquare className="w-4 h-4 text-muted-foreground" />
          <p className="text-sm font-medium">
            Additional comments <span className="text-muted-foreground">(optional)</span>
          </p>
        </div>
        <Textarea
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="What did you like or dislike about this eval prompt?"
          className="min-h-[80px]"
        />
      </div>

      <Button
        onClick={handleSubmit}
        disabled={submitting || rating === 0}
        className="w-full"
      >
        {submitting ? (
          <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Submitting...</>
        ) : (
          'Submit Feedback'
        )}
      </Button>

      {rating === 0 && (
        <p className="text-xs text-muted-foreground text-center">
          Select a star rating to continue
        </p>
      )}
    </div>
  );
}
