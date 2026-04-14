import mongoose from "mongoose";

const scoreSchema = new mongoose.Schema(
  {
    suspectId: { type: mongoose.Schema.Types.ObjectId, ref: "Suspect" },
    similarity: { type: Number, required: true },
    name: { type: String },
  },
  { _id: false }
);

const matchResultSchema = new mongoose.Schema(
  {
    query_video_path: { type: String, required: true },
    scores: { type: [scoreSchema], default: [] },
    top_suspect: { type: mongoose.Schema.Types.ObjectId, ref: "Suspect" },
  },
  { timestamps: true }
);

export const MatchResult = mongoose.model("MatchResult", matchResultSchema);
