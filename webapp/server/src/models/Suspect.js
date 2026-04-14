import mongoose from "mongoose";

const suspectSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
    signature: { type: [Number], required: true },
    video_path: { type: String, required: true },
  },
  { timestamps: true }
);

export const Suspect = mongoose.model("Suspect", suspectSchema);
