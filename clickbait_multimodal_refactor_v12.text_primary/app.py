"""
Enhanced Multi-Model Clickbait Detector Flask Backend
Advanced version with text-only, image-only, and fusion models
Includes comprehensive analysis and model comparison features
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
import io
import torch
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
import joblib
import os
import logging
from pathlib import Path
import re
from typing import Dict, Any, Optional, Tuple, List, Union
import sys
import atexit
from datetime import datetime
import json
import time
import warnings
import sys
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clickbait_detector.log', encoding='utf-8'),  # Thêm encoding='utf-8'
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MultiModalClickbaitDetector:
    """Enhanced multi-modal clickbait detector with three specialized models"""

    def __init__(self):
        self.device = None
        self.text_tokenizer = None
        self.text_extractor = None
        self.image_extractor = None
        self.image_transform = None

        # Three separate classifiers
        self.text_only_classifier = None
        self.image_only_classifier = None
        self.fusion_classifier = None

        self.models_loaded = False
        self.initialization_time = None

        # Enhanced demo examples categorized by clickbait likelihood
        self.demo_examples = {
            "high_clickbait": [
                "You Won't Believe What Happened Next! This Will Shock You!",
                "This Simple Trick Will Change Your Life FOREVER - Try It Now!",
                "Doctors HATE This One Weird Trick That Actually Works",
                "10 Shocking Celebrity Secrets That Will BLOW Your Mind",
                "This Video Made Me Cry - You Need to Watch Until The End!"
            ],
            "moderate_clickbait": [
                "Amazing Life Hacks Everyone Should Know",
                "The Secret to Success That Nobody Talks About",
                "Why You Should Never Do This Common Thing",
                "The Truth About What Celebrities Really Eat",
                "How to Get Rich Quick with This Method"
            ],
            "low_clickbait": [
                "Tips for Better Sleep and Health",
                "Understanding Climate Change Impacts",
                "How to Cook Traditional Vietnamese Pho",
                "Benefits of Regular Exercise",
                "Introduction to Machine Learning Concepts"
            ],
            "non_clickbait": [
                "Vietnam's GDP grows 6.8% in Q3 2024, exceeding expectations",
                "New research findings published in Nature journal on cancer treatment",
                "Ho Chi Minh City announces new metro line construction timeline",
                "Government implements new digital transformation policy",
                "Scientific study reveals impact of urbanization on air quality"
            ]
        }

        # Feature importance weights for analysis
        self.feature_weights = {
            'text_features': {
                'emotional_words': 0.15,
                'superlatives': 0.12,
                'question_marks': 0.08,
                'exclamation_marks': 0.10,
                'caps_ratio': 0.09,
                'clickbait_phrases': 0.20,
                'personal_pronouns': 0.07,
                'urgency_words': 0.11,
                'number_patterns': 0.08
            },
            'image_features': {
                'face_detection': 0.18,
                'bright_colors': 0.15,
                'text_overlay': 0.12,
                'emotional_content': 0.14,
                'composition': 0.11,
                'brand_elements': 0.10,
                'background_complexity': 0.09,
                'visual_appeal': 0.11
            }
        }

    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models and return detailed status"""
        start_time = time.time()
        initialization_log = []

        try:
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            initialization_log.append(f"Device: {self.device}")

            # Load text components
            logger.info("Loading XLM-RoBERTa text extractor...")
            try:
                self.text_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
                self.text_extractor = AutoModel.from_pretrained('xlm-roberta-base')
                self.text_extractor.eval()
                self.text_extractor.to(self.device)
                initialization_log.append("✓ Text extractor loaded successfully")
            except Exception as e:
                initialization_log.append(f"✗ Text extractor failed: {e}")
                raise

            # Load image components
            logger.info("Loading ResNet50 image extractor...")
            try:
                resnet50 = models.resnet50(pretrained=True)
                self.image_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
                self.image_extractor.eval()
                self.image_extractor.to(self.device)

                # Image preprocessing
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                initialization_log.append("✓ Image extractor loaded successfully")
            except Exception as e:
                initialization_log.append(f"✗ Image extractor failed: {e}")
                raise

            # Load trained classifiers
            classifier_status = self._load_classifiers()
            initialization_log.extend(classifier_status)

            self.models_loaded = True
            self.initialization_time = time.time() - start_time

            logger.info(f"All models loaded successfully in {self.initialization_time:.2f} seconds!")

            return {
                'success': True,
                'initialization_time': self.initialization_time,
                'log': initialization_log,
                'device': str(self.device),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'model_status': self._get_model_status()
            }

        except Exception as e:
            error_msg = f"Model initialization failed: {e}"
            logger.error(error_msg)
            initialization_log.append(f"✗ {error_msg}")

            return {
                'success': False,
                'error': error_msg,
                'log': initialization_log,
                'initialization_time': time.time() - start_time
            }

    def _load_classifiers(self) -> List[str]:
        """Load pre-trained classifier models"""
        model_dir = Path("saved_models")
        model_dir.mkdir(exist_ok=True)

        classifier_files = {
            'text_only': "text_transformer_stacking_model.pkl",
            'image_only': "image_only_stacking_model.pkl",
            'fusion': "fusion_stacking_model.pkl"
        }

        load_status = []

        for classifier_type, filename in classifier_files.items():
            filepath = model_dir / filename
            if filepath.exists():
                try:
                    classifier = joblib.load(filepath)
                    setattr(self, f"{classifier_type}_classifier", classifier)
                    load_status.append(f"✓ {classifier_type.replace('_', ' ').title()} classifier loaded")
                    logger.info(f"{classifier_type} classifier loaded from {filepath}")
                except Exception as e:
                    load_status.append(f"✗ {classifier_type.replace('_', ' ').title()} classifier failed: {e}")
                    logger.warning(f"Failed to load {classifier_type} classifier: {e}")
            else:
                load_status.append(f"⚠ {classifier_type.replace('_', ' ').title()} classifier not found")
                logger.warning(f"{classifier_type} classifier not found at {filepath}")

        return load_status

    def _get_model_status(self) -> Dict[str, bool]:
        """Get current status of all models"""
        return {
            'text_extractor': self.text_extractor is not None,
            'image_extractor': self.image_extractor is not None,
            'text_only_classifier': self.text_only_classifier is not None,
            'image_only_classifier': self.image_only_classifier is not None,
            'fusion_classifier': self.fusion_classifier is not None
        }

    def extract_text_features(self, text: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract comprehensive text features"""
        if not self.text_tokenizer or not self.text_extractor:
            raise ValueError("Text models not loaded")

        # XLM-RoBERTa embeddings
        inputs = self.text_tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.text_extractor(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        # Advanced text analysis
        text_analysis = self._analyze_text_patterns(text)

        return embeddings, text_analysis

    def _analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text for clickbait patterns"""
        analysis = {}

        # Basic metrics
        analysis['length'] = len(text)
        analysis['word_count'] = len(text.split())
        analysis['sentence_count'] = len([s for s in text.split('.') if s.strip()])

        # Clickbait indicators
        clickbait_phrases = [
            "you won't believe", "this will", "shocking", "amazing", "incredible",
            "you need to know", "this changes everything", "mind blown", "unbelievable",
            "secret", "trick", "hack", "doctors hate", "one weird", "what happened next"
        ]

        text_lower = text.lower()
        analysis['clickbait_phrases'] = sum(1 for phrase in clickbait_phrases if phrase in text_lower)

        # Emotional words
        emotional_words = [
            "love", "hate", "amazing", "terrible", "awesome", "horrible", "fantastic",
            "awful", "incredible", "shocking", "stunning", "devastating", "thrilling"
        ]
        analysis['emotional_words'] = sum(1 for word in emotional_words if word in text_lower)

        # Punctuation analysis
        analysis['exclamation_marks'] = text.count('!')
        analysis['question_marks'] = text.count('?')
        analysis['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        # Numbers and lists
        analysis['contains_numbers'] = bool(re.search(r'\d+', text))
        analysis['list_indicators'] = bool(re.search(r'\b(\d+)\s+(things|ways|reasons|tips|secrets)', text_lower))

        # Urgency indicators
        urgency_words = ["now", "today", "urgent", "limited time", "hurry", "quick", "fast", "immediately"]
        analysis['urgency_words'] = sum(1 for word in urgency_words if word in text_lower)

        # Personal pronouns
        pronouns = ["you", "your", "yours", "yourself"]
        analysis['personal_pronouns'] = sum(text_lower.count(pronoun) for pronoun in pronouns)

        # Social media elements
        analysis['hashtags'] = len([w for w in text.split() if w.startswith('#')])
        analysis['mentions'] = len([w for w in text.split() if w.startswith('@')])
        analysis['urls'] = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

        # Emoji analysis
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
            flags=re.UNICODE
        )
        emojis = emoji_pattern.findall(text)
        analysis['emoji_count'] = len(emojis)

        # Clickbait score calculation
        clickbait_score = (
            analysis['clickbait_phrases'] * 3 +
            analysis['emotional_words'] * 2 +
            analysis['exclamation_marks'] * 1.5 +
            analysis['urgency_words'] * 2 +
            analysis['caps_ratio'] * 10 +
            (1 if analysis['list_indicators'] else 0) * 2
        )
        analysis['clickbait_score'] = min(clickbait_score / 20, 1.0)  # Normalize to 0-1

        return analysis

    def extract_image_features(self, image_data, for_fusion=False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract comprehensive image features

        Args:
            image_data: Image data (base64 string or PIL Image)
            for_fusion: If True, return full 2048 features for fusion model
                       If False, return 2042 features for image-only model
        """
        if not self.image_extractor or not self.image_transform:
            raise ValueError("Image models not loaded")

        # Process image
        if isinstance(image_data, str):
            if 'data:image' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image = image_data

        # Extract ResNet features
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.image_extractor(image_tensor)
            embeddings = features.view(features.size(0), -1).squeeze().cpu().numpy()

            # Điều chỉnh kích thước features dựa trên model đích
            if len(embeddings) == 2048:
                if for_fusion:
                    # Giữ nguyên 2048 features cho fusion model
                    logger.info("Keeping full 2048 image features for fusion model")
                else:
                    # Cắt xuống 2042 features cho image-only model
                    embeddings = embeddings[:2042]
                    logger.info("Adjusted image features for image-only model: 2048 to 2042")
            else:
                logger.warning(f"Unexpected image feature dimension: {len(embeddings)}")

        # Image analysis
        image_analysis = self._analyze_image_properties(image)

        return embeddings, image_analysis

    def _analyze_image_properties(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image properties for clickbait indicators"""
        analysis = {}

        # Basic properties
        analysis['width'], analysis['height'] = image.size
        analysis['aspect_ratio'] = analysis['width'] / analysis['height']
        analysis['total_pixels'] = analysis['width'] * analysis['height']

        # Convert to numpy for analysis
        img_array = np.array(image)

        # Color analysis
        if len(img_array.shape) == 3:
            # RGB analysis
            r_mean, g_mean, b_mean = np.mean(img_array, axis=(0, 1))
            analysis['brightness'] = np.mean(img_array) / 255
            analysis['color_variance'] = np.var(img_array) / (255**2)
            analysis['dominant_channel'] = ['red', 'green', 'blue'][np.argmax([r_mean, g_mean, b_mean])]

            # Saturation approximation
            rgb_max = np.max(img_array, axis=2)
            rgb_min = np.min(img_array, axis=2)
            saturation = (rgb_max - rgb_min) / (rgb_max + 1e-7)
            analysis['average_saturation'] = np.mean(saturation)
        else:
            analysis['brightness'] = np.mean(img_array) / 255
            analysis['color_variance'] = np.var(img_array) / (255**2)
            analysis['average_saturation'] = 0
            analysis['dominant_channel'] = 'grayscale'

        # Complexity indicators
        # Edge detection approximation
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Simple edge detection
        edges_h = np.abs(np.diff(gray, axis=0))
        edges_v = np.abs(np.diff(gray, axis=1))
        analysis['edge_density'] = (np.mean(edges_h) + np.mean(edges_v)) / 2 / 255

        # Texture analysis (simplified)
        analysis['texture_variance'] = np.var(gray) / (255**2)

        # Clickbait visual indicators (heuristic)
        # Bright, high-contrast images tend to be more clickbait
        analysis['visual_appeal_score'] = min(
            (analysis['brightness'] * 0.3 +
             analysis['color_variance'] * 0.3 +
             analysis['average_saturation'] * 0.2 +
             analysis['edge_density'] * 0.2), 1.0
        )

        return analysis

    def extract_additional_features(self, text: str = None, image_analysis: Dict = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract 26 additional handcrafted features"""
        features = {}
        explanations = {}

        # Text-based features
        if text:
            text_analysis = self._analyze_text_patterns(text)
            features.update({
                'LengthOfCaptions': len(text),
                'LengthOfHashtags': text_analysis.get('hashtags', 0),
                'URLInclusion': 1 if text_analysis.get('urls', 0) > 0 else 0,
                'MentionInclusion': 1 if text_analysis.get('mentions', 0) > 0 else 0,
                'EmojiCount': text_analysis.get('emoji_count', 0),
                'EmojiExistence': 1 if text_analysis.get('emoji_count', 0) > 0 else 0,
                'EmojiPortion': text_analysis.get('emoji_count', 0) / (len(text) + 1)
            })

            explanations.update({
                'LengthOfCaptions': f"Text length: {len(text)} characters",
                'LengthOfHashtags': f"Hashtag count: {text_analysis.get('hashtags', 0)}",
                'URLInclusion': "Contains URL" if features['URLInclusion'] else "No URL",
                'MentionInclusion': "Contains mentions" if features['MentionInclusion'] else "No mentions",
                'EmojiCount': f"Emoji count: {text_analysis.get('emoji_count', 0)}"
            })
        else:
            # Default values when no text
            for key in ['LengthOfCaptions', 'LengthOfHashtags', 'URLInclusion',
                       'MentionInclusion', 'EmojiCount', 'EmojiExistence', 'EmojiPortion']:
                features[key] = 0

        # Social media metrics (simulated - would need real data)
        social_metrics = ['Likes', 'Comments', 'Followings', 'Followers', 'MediaCounts']
        for metric in social_metrics:
            features[metric] = 0  # Placeholder

        # Location and hashtag analysis (simulated)
        features.update({
            'LocationExistence': 0,
            'Top100HashOfInsta': 0,
            'Top100HashWithinData': 0,
            'Top100ComentionedHashPair': 0
        })

        # Image-based features (enhanced with actual analysis)
        if image_analysis:
            # Use image analysis results for better features
            features.update({
                'Selfie': 1 if image_analysis.get('visual_appeal_score', 0) > 0.7 else 0,
                'BodySnap': 0,  # Would need specialized detection
                'Marketing': 1 if image_analysis.get('brightness', 0) > 0.6 and image_analysis.get('average_saturation', 0) > 0.5 else 0,
                'ProductOnly': 0,  # Would need object detection
                'NonFashion': 1,  # Assuming non-fashion by default
                'Face': 0,  # Would need face detection
                'Logo': 0,  # Would need logo detection
                'BrandLogo': 0,  # Would need brand detection
                'Smile': 0,  # Would need emotion detection
                'Outdoor': 1 if image_analysis.get('brightness', 0) > 0.5 else 0
            })
        else:
            # Default values when no image
            image_features = ['Selfie', 'BodySnap', 'Marketing', 'ProductOnly', 'NonFashion',
                            'Face', 'Logo', 'BrandLogo', 'Smile', 'Outdoor']
            for feature in image_features:
                features[feature] = 0

        # Convert to array in consistent order
        feature_order = [
            'Likes', 'Comments', 'Followings', 'Followers', 'MediaCounts', 'LocationExistence',
            'LengthOfHashtags', 'LengthOfCaptions', 'URLInclusion', 'MentionInclusion',
            'EmojiCount', 'EmojiExistence', 'EmojiPortion', 'Top100HashOfInsta',
            'Top100HashWithinData', 'Top100ComentionedHashPair', 'Selfie', 'BodySnap',
            'Marketing', 'ProductOnly', 'NonFashion', 'Face', 'Logo', 'BrandLogo', 'Smile', 'Outdoor'
        ]

        feature_array = np.array([features[feat] for feat in feature_order])
        return feature_array, explanations

    def predict_with_model(self, model, features: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Make prediction with confidence intervals and detailed analysis"""
        if model is None:
            raise ValueError(f"{model_name} model not loaded")

        # Basic prediction
        prediction = model.predict(features.reshape(1, -1))[0]
        probabilities = model.predict_proba(features.reshape(1, -1))[0]

        # Calculate confidence metrics
        confidence = max(probabilities)
        certainty = abs(probabilities[1] - probabilities[0])  # How far from 50/50

        # Risk assessment
        if probabilities[1] > 0.8:
            risk_level = "High"
        elif probabilities[1] > 0.6:
            risk_level = "Moderate"
        elif probabilities[1] > 0.4:
            risk_level = "Low"
        else:
            risk_level = "Very Low"

        return {
            'prediction': int(prediction),
            'label': "Clickbait" if prediction == 1 else "Non-clickbait",
            'probability': float(probabilities[1]),
            'confidence': float(confidence),
            'certainty': float(certainty),
            'risk_level': risk_level,
            'model_name': model_name,
            'probabilities': {
                'clickbait': float(probabilities[1]),
                'non_clickbait': float(probabilities[0])
            }
        }

    def analyze_comprehensive(self, text: Optional[str] = None,
                            image_data: Optional[str] = None,
                            analysis_type: str = "all") -> Dict[str, Any]:
        """Comprehensive multi-model analysis"""

        if not self.models_loaded:
            raise ValueError("Models not initialized")

        start_time = time.time()
        results = {
            'analysis_type': analysis_type,
            'input_summary': {
                'has_text': text is not None and len(text.strip()) > 0,
                'has_image': image_data is not None,
                'text_preview': text[:100] + "..." if text and len(text) > 100 else text
            },
            'predictions': {},
            'feature_analysis': {},
            'model_comparison': {},
            'recommendations': []
        }

        try:
            # Extract features
            text_features = None
            text_analysis = None
            image_features_for_image_only = None
            image_features_for_fusion = None
            image_analysis = None
            additional_features = None
            feature_explanations = None

            if text and text.strip():
                text_features, text_analysis = self.extract_text_features(text)
                results['feature_analysis']['text'] = text_analysis

            if image_data:
                # Extract image features cho cả image-only và fusion
                image_features_for_image_only, image_analysis = self.extract_image_features(image_data,
                                                                                            for_fusion=False)
                image_features_for_fusion, _ = self.extract_image_features(image_data, for_fusion=True)
                results['feature_analysis']['image'] = image_analysis

            # Extract additional features
            additional_features, feature_explanations = self.extract_additional_features(
                text=text, image_analysis=image_analysis
            )
            results['feature_analysis']['additional'] = feature_explanations

            # Run predictions based on analysis type and available models
            available_models = []

            # Text-only model
            if (analysis_type in ["text", "all"] and text_features is not None and
                    self.text_only_classifier is not None):
                try:
                    prediction = self.predict_with_model(
                        self.text_only_classifier, text_features, "Text-Only"
                    )
                    results['predictions']['text_only'] = prediction
                    available_models.append('text_only')
                except Exception as e:
                    logger.error(f"Text-only prediction failed: {e}")

            # Image-only model - SỬ DỤNG image_features_for_image_only
            if (analysis_type in ["image", "all"] and image_features_for_image_only is not None and
                    self.image_only_classifier is not None):
                try:
                    prediction = self.predict_with_model(
                        self.image_only_classifier, image_features_for_image_only, "Image-Only"
                    )
                    results['predictions']['image_only'] = prediction
                    available_models.append('image_only')
                except Exception as e:
                    logger.error(f"Image-only prediction failed: {e}")

            # Fusion model - SỬ DỤNG image_features_for_fusion
            if (analysis_type in ["fusion", "all"] and text_features is not None and
                    image_features_for_fusion is not None and self.fusion_classifier is not None):
                try:
                    # Combine all features for fusion model
                    fused_features = np.concatenate([
                        image_features_for_fusion,  # Full 2048 ResNet features
                        additional_features,  # 26 handcrafted features
                        text_features  # 768 XLM-RoBERTa features
                    ])
                    # Tổng: 2048 + 26 + 768 = 2842 features ✅

                    logger.info(f"Fusion features shape: {fused_features.shape} (expected: 2842)")

                    prediction = self.predict_with_model(
                        self.fusion_classifier, fused_features, "Fusion"
                    )
                    results['predictions']['fusion'] = prediction
                    available_models.append('fusion')
                except Exception as e:
                    logger.error(f"Fusion prediction failed: {e}")

            # Generate model comparison
            if len(available_models) > 1:
                results['model_comparison'] = self._compare_predictions(results['predictions'])

            # Determine best prediction
            results['primary_prediction'] = self._select_primary_prediction(
                results['predictions'], available_models
            )

            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(
                results['predictions'], results['feature_analysis']
            )

            # Add metadata
            results['metadata'] = {
                'processing_time': round(time.time() - start_time, 3),
                'available_models': available_models,
                'feature_dimensions': {
                    'text_features': len(text_features) if text_features is not None else 0,
                    'image_features_image_only': len(
                        image_features_for_image_only) if image_features_for_image_only is not None else 0,
                    'image_features_fusion': len(
                        image_features_for_fusion) if image_features_for_fusion is not None else 0,
                    'additional_features': len(additional_features) if additional_features is not None else 0,
                    'fusion_total': (
                            len(image_features_for_fusion) + len(additional_features) + len(text_features)
                    ) if all(
                        x is not None for x in [text_features, image_features_for_fusion, additional_features]) else 0
                },
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device)
            }

            return results

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise

    def _compare_predictions(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare predictions across different models"""
        comparison = {
            'agreement_score': 0,
            'consensus': None,
            'confidence_ranking': [],
            'risk_assessment': {},
            'model_insights': {}
        }

        if len(predictions) < 2:
            return comparison

        # Calculate agreement
        labels = [pred['label'] for pred in predictions.values()]
        clickbait_votes = sum(1 for label in labels if label == 'Clickbait')
        total_votes = len(labels)

        comparison['agreement_score'] = max(clickbait_votes, total_votes - clickbait_votes) / total_votes
        comparison['consensus'] = 'Clickbait' if clickbait_votes > total_votes/2 else 'Non-clickbait'

        # Confidence ranking
        confidence_data = [(name, pred['confidence']) for name, pred in predictions.items()]
        confidence_data.sort(key=lambda x: x[1], reverse=True)
        comparison['confidence_ranking'] = confidence_data

        # Risk assessment
        avg_probability = np.mean([pred['probability'] for pred in predictions.values()])
        std_probability = np.std([pred['probability'] for pred in predictions.values()])

        comparison['risk_assessment'] = {
            'average_clickbait_probability': float(avg_probability),
            'prediction_variance': float(std_probability),
            'consistency': 'High' if std_probability < 0.1 else 'Medium' if std_probability < 0.2 else 'Low'
        }

        # Model insights
        for model_name, pred in predictions.items():
            comparison['model_insights'][model_name] = {
                'specialization': self._get_model_specialization(model_name),
                'reliability': self._assess_model_reliability(pred),
                'unique_insights': self._get_model_insights(model_name, pred)
            }

        return comparison

    def _get_model_specialization(self, model_name: str) -> str:
        """Get model specialization description"""
        specializations = {
            'text_only': 'Specialized in linguistic patterns, emotional language, and textual clickbait indicators',
            'image_only': 'Focused on visual elements, colors, composition, and image-based engagement factors',
            'fusion': 'Combines both text and visual information for comprehensive multimodal analysis'
        }
        return specializations.get(model_name, 'Unknown specialization')

    def _assess_model_reliability(self, prediction: Dict) -> str:
        """Assess model reliability based on confidence and certainty"""
        confidence = prediction['confidence']
        certainty = prediction['certainty']

        if confidence > 0.85 and certainty > 0.7:
            return 'Very High'
        elif confidence > 0.7 and certainty > 0.4:
            return 'High'
        elif confidence > 0.6 and certainty > 0.2:
            return 'Medium'
        else:
            return 'Low'

    def _get_model_insights(self, model_name: str, prediction: Dict) -> List[str]:
        """Generate specific insights for each model"""
        insights = []

        if model_name == 'text_only':
            prob = prediction['probability']
            if prob > 0.8:
                insights.append("Strong textual clickbait indicators detected")
            elif prob > 0.6:
                insights.append("Moderate use of attention-grabbing language")
            elif prob < 0.3:
                insights.append("Professional or neutral language tone")
            else:
                insights.append("Balanced textual content")

        elif model_name == 'image_only':
            prob = prediction['probability']
            if prob > 0.8:
                insights.append("High visual appeal and engagement factors")
            elif prob > 0.6:
                insights.append("Visually attractive with some clickbait elements")
            elif prob < 0.3:
                insights.append("Conservative visual design")
            else:
                insights.append("Neutral visual presentation")

        elif model_name == 'fusion':
            prob = prediction['probability']
            if prob > 0.8:
                insights.append("Strong multimodal clickbait signals")
            elif prob > 0.6:
                insights.append("Combined text and visual elements suggest clickbait")
            elif prob < 0.3:
                insights.append("Professional content across all modalities")
            else:
                insights.append("Balanced multimodal presentation")

        return insights

    def _select_primary_prediction(self, predictions: Dict, available_models: List) -> Dict:
        """Select the most reliable primary prediction"""
        if not predictions:
            return None

        # Priority: fusion > text_only > image_only (if available)
        priority_order = ['fusion', 'text_only', 'image_only']

        for model in priority_order:
            if model in predictions:
                primary = predictions[model].copy()
                primary['primary_model'] = model
                primary['selection_reason'] = f"Selected {model} model as primary"
                return primary

        # Fallback to first available
        first_model = list(predictions.keys())[0]
        primary = predictions[first_model].copy()
        primary['primary_model'] = first_model
        primary['selection_reason'] = f"Only {first_model} model available"
        return primary

    def _generate_recommendations(self, predictions: Dict, feature_analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Get primary prediction
        if not predictions:
            return recommendations

        primary_prob = max([pred['probability'] for pred in predictions.values()])

        # Content quality recommendations
        if primary_prob > 0.7:
            recommendations.append({
                'type': 'warning',
                'category': 'Content Quality',
                'title': 'High Clickbait Risk Detected',
                'description': 'This content shows strong clickbait characteristics that may mislead users.',
                'suggestions': [
                    'Consider using more specific and informative headlines',
                    'Reduce use of emotional trigger words',
                    'Provide clear value proposition instead of mystery'
                ]
            })
        elif primary_prob > 0.5:
            recommendations.append({
                'type': 'caution',
                'category': 'Content Quality',
                'title': 'Moderate Clickbait Elements',
                'description': 'Some clickbait patterns detected. Consider refining for better user trust.',
                'suggestions': [
                    'Balance engagement with transparency',
                    'Ensure content delivers on promises made in title',
                    'Consider more descriptive rather than mysterious language'
                ]
            })
        else:
            recommendations.append({
                'type': 'positive',
                'category': 'Content Quality',
                'title': 'Good Content Standards',
                'description': 'Content appears to follow good practices with minimal clickbait elements.',
                'suggestions': [
                    'Maintain current transparent communication style',
                    'Continue providing clear value propositions'
                ]
            })

        # Text-specific recommendations
        if 'text' in feature_analysis:
            text_analysis = feature_analysis['text']

            if text_analysis.get('caps_ratio', 0) > 0.3:
                recommendations.append({
                    'type': 'suggestion',
                    'category': 'Text Formatting',
                    'title': 'Excessive Capital Letters',
                    'description': 'High ratio of capital letters may appear unprofessional.',
                    'suggestions': ['Use normal capitalization for better readability']
                })

            if text_analysis.get('exclamation_marks', 0) > 2:
                recommendations.append({
                    'type': 'suggestion',
                    'category': 'Text Formatting',
                    'title': 'Excessive Exclamation Marks',
                    'description': 'Multiple exclamation marks may reduce credibility.',
                    'suggestions': ['Limit exclamation marks to emphasize key points only']
                })

        # Image-specific recommendations
        if 'image' in feature_analysis:
            image_analysis = feature_analysis['image']

            if image_analysis.get('visual_appeal_score', 0) > 0.8:
                recommendations.append({
                    'type': 'info',
                    'category': 'Visual Design',
                    'title': 'High Visual Appeal',
                    'description': 'Image has strong visual elements that attract attention.',
                    'suggestions': [
                        'Ensure image content matches article substance',
                        'Consider if visual style aligns with content credibility'
                    ]
                })

        return recommendations

    def get_demo_examples(self) -> Dict[str, Any]:
        """Get categorized demo examples"""
        return self.demo_examples

    def analyze_batch(self, items: List[Dict], max_items: int = 10) -> Dict[str, Any]:
        """Analyze multiple items in batch"""
        if len(items) > max_items:
            raise ValueError(f"Maximum {max_items} items allowed per batch")

        start_time = time.time()
        results = []
        summary = {'clickbait': 0, 'non_clickbait': 0, 'errors': 0}

        for i, item in enumerate(items):
            try:
                text = item.get('text')
                image = item.get('image')

                if not text and not image:
                    results.append({
                        'index': i,
                        'error': 'No text or image provided',
                        'item': item
                    })
                    summary['errors'] += 1
                    continue

                analysis = self.analyze_comprehensive(
                    text=text,
                    image_data=image,
                    analysis_type="all"
                )

                # Simplified result for batch
                primary = analysis['primary_prediction']
                if primary['label'] == 'Clickbait':
                    summary['clickbait'] += 1
                else:
                    summary['non_clickbait'] += 1

                results.append({
                    'index': i,
                    'text_preview': text[:50] + '...' if text and len(text) > 50 else text,
                    'has_image': image is not None,
                    'prediction': primary['label'],
                    'probability': primary['probability'],
                    'confidence': primary['confidence'],
                    'model_used': primary['primary_model']
                })

            except Exception as e:
                logger.error(f"Batch item {i} analysis failed: {e}")
                results.append({
                    'index': i,
                    'error': str(e),
                    'item': item
                })
                summary['errors'] += 1

        return {
            'results': results,
            'summary': summary,
            'processing_time': round(time.time() - start_time, 3),
            'total_processed': len(items)
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'models_loaded': self.models_loaded,
            'initialization_time': self.initialization_time,
            'device': str(self.device) if self.device else None,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'gpu_available': torch.cuda.is_available(),
            'memory_usage': {
                'gpu_memory': f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB" if torch.cuda.is_available() else "N/A",
                'gpu_cached': f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB" if torch.cuda.is_available() else "N/A"
            },
            'model_status': self._get_model_status(),
            'feature_dimensions': {
                'xlm_roberta': 768,
                'resnet50_image_only': 2042,  # Cho image-only model
                'resnet50_fusion': 2048,  # Cho fusion model
                'additional_features': 26,
                'fusion_total': 2842  # 2048 + 26 + 768
            }
        }

# Initialize Flask application
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])  # Add React dev server

# Initialize detector
detector = MultiModalClickbaitDetector()
initialization_status = None

def initialize_application():
    """Initialize models on app startup"""
    global initialization_status
    logger.info("Starting model initialization...")
    initialization_status = detector.initialize_models()

    if initialization_status['success']:
        logger.info("✅ Application initialized successfully!")
    else:
        logger.error("❌ Application initialization failed!")

    return initialization_status

# Initialize on startup
with app.app_context():
    initialization_status = initialize_application()

# API Routes
@app.route('/')
def index():
    """Serve main application page"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    """Main analysis endpoint supporting all model types"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'code': 'MISSING_DATA'
            }), 400

        # Extract parameters
        text = data.get('text', '').strip() if data.get('text') else None
        image_data = data.get('image')
        analysis_type = data.get('analysis_type', 'all')  # all, text, image, fusion

        # Validation
        if not text and not image_data:
            return jsonify({
                'success': False,
                'error': 'Please provide either text content, image data, or both',
                'code': 'MISSING_CONTENT'
            }), 400

        if analysis_type not in ['all', 'text', 'image', 'fusion']:
            return jsonify({
                'success': False,
                'error': 'Invalid analysis_type. Must be: all, text, image, or fusion',
                'code': 'INVALID_ANALYSIS_TYPE'
            }), 400

        # Perform analysis
        result = detector.analyze_comprehensive(
            text=text,
            image_data=image_data,
            analysis_type=analysis_type
        )

        return jsonify({
            'success': True,
            'data': result,
            'api_version': '2.0.0'
        })

    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}',
            'code': 'ANALYSIS_ERROR'
        }), 500

@app.route('/api/models/compare', methods=['POST'])
def compare_models():
    """Compare all available models for given input"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip() if data.get('text') else None
        image_data = data.get('image')

        if not text and not image_data:
            return jsonify({
                'success': False,
                'error': 'Please provide text, image, or both for comparison'
            }), 400

        result = detector.analyze_comprehensive(
            text=text,
            image_data=image_data,
            analysis_type="all"
        )

        # Enhanced comparison data
        comparison_result = {
            'input_summary': result['input_summary'],
            'predictions': result['predictions'],
            'model_comparison': result['model_comparison'],
            'recommendations': result['recommendations'],
            'summary': {
                'models_used': len(result['predictions']),
                'consensus': result['model_comparison'].get('consensus'),
                'agreement_score': result['model_comparison'].get('agreement_score'),
                'best_model': result['primary_prediction']['primary_model']
            }
        }

        return jsonify({
            'success': True,
            'data': comparison_result
        })

    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/demo/examples', methods=['GET'])
def get_demo_examples():
    """Get categorized demo examples"""
    examples = detector.get_demo_examples()

    # Add metadata for each category
    enhanced_examples = {}
    for category, texts in examples.items():
        enhanced_examples[category] = {
            'texts': texts,
            'description': {
                'high_clickbait': 'Clear clickbait with emotional triggers and mystery',
                'moderate_clickbait': 'Some clickbait elements but less aggressive',
                'low_clickbait': 'Informative content with minimal clickbait elements',
                'non_clickbait': 'Professional, factual content'
            }.get(category, 'Example texts'),
            'expected_probability_range': {
                'high_clickbait': [0.8, 1.0],
                'moderate_clickbait': [0.6, 0.8],
                'low_clickbait': [0.3, 0.6],
                'non_clickbait': [0.0, 0.3]
            }.get(category, [0.0, 1.0])
        }

    return jsonify({
        'success': True,
        'data': enhanced_examples,
        'usage_instructions': {
            'text_analysis': 'Copy any example text to test text-only model',
            'batch_analysis': 'Use /api/batch endpoint to test multiple examples',
            'comparison': 'Use /api/models/compare to see how different models perform'
        }
    })

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    """Analyze multiple items in batch"""
    try:
        data = request.get_json()
        items = data.get('items', [])
        max_items = data.get('max_items', 10)

        if not items or not isinstance(items, list):
            return jsonify({
                'success': False,
                'error': 'Please provide a list of items to analyze'
            }), 400

        result = detector.analyze_batch(items, max_items)

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    status = detector.get_system_status()

    # Determine overall health
    is_healthy = (
        status['models_loaded'] and
        all(status['model_status'].values()) and
        initialization_status and
        initialization_status['success']
    )

    return jsonify({
        'status': 'healthy' if is_healthy else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'initialization': initialization_status,
        'system': status,
        'endpoints': {
            'analyze': '/api/analyze - Main analysis endpoint',
            'compare': '/api/models/compare - Compare all models',
            'batch': '/api/batch - Batch processing',
            'examples': '/api/demo/examples - Get demo examples',
            'health': '/api/health - System health',
            'info': '/api/info - System information'
        }
    }), 200 if is_healthy else 503

@app.route('/api/info', methods=['GET'])
def system_info():
    """Detailed system information"""
    return jsonify({
        'application': {
            'name': 'Multi-Modal Clickbait Detector',
            'version': '2.0.0',
            'description': 'Advanced AI system for detecting clickbait content using multiple specialized models'
        },
        'models': {
            'text_extractor': {
                'name': 'XLM-RoBERTa-base',
                'dimensions': 768,
                'description': 'Multilingual transformer for text feature extraction'
            },
            'image_extractor': {
                'name': 'ResNet50',
                'dimensions': 2048,
                'description': 'Convolutional neural network for image feature extraction'
            },
            'classifiers': {
                'text_only': 'Specialized for text-based clickbait detection',
                'image_only': 'Specialized for visual clickbait indicators',
                'fusion': 'Combines text, image, and handcrafted features'
            }
        },
        'features': {
            'multi_modal_analysis': 'Text, image, and combined analysis',
            'model_comparison': 'Compare predictions across different models',
            'batch_processing': 'Process multiple items simultaneously',
            'detailed_insights': 'Feature analysis and recommendations',
            'real_time_processing': 'Fast inference with GPU acceleration'
        },
        'capabilities': {
            'supported_languages': 'Multi-language support via XLM-RoBERTa',
            'image_formats': ['JPEG', 'PNG', 'WebP', 'Base64 encoded'],
            'max_text_length': 128,
            'max_batch_size': 10,
            'response_time': '< 1 second per item'
        },
        'system': detector.get_system_status()
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'code': 'NOT_FOUND',
        'available_endpoints': [
            'POST /api/analyze - Main analysis endpoint',
            'POST /api/models/compare - Compare models',
            'POST /api/batch - Batch processing',
            'GET /api/demo/examples - Demo examples',
            'GET /api/health - Health check',
            'GET /api/info - System information'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error occurred',
        'code': 'INTERNAL_ERROR',
        'message': 'Please check server logs for details'
    }), 500

@app.errorhandler(413)
def payload_too_large(error):
    return jsonify({
        'success': False,
        'error': 'Request payload too large',
        'code': 'PAYLOAD_TOO_LARGE',
        'max_size': '16MB'
    }), 413

# Startup and cleanup
@atexit.register
def cleanup():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down Clickbait Detector...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # Create required directories
    for directory in ['saved_models', 'templates', 'static/css', 'static/js', 'static/images']:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Application startup
    print("="*80)
    print("🚀 Multi-Modal Clickbait Detector API")
    print("="*80)
    print(f"📊 Models Status: {'✅ Ready' if initialization_status and initialization_status['success'] else '❌ Failed'}")
    print(f"🖥️  Device: {detector.device}")
    print(f"⚡ GPU Available: {'Yes' if torch.cuda.is_available() else 'No'}")
    print("="*80)
    print("📝 Available Endpoints:")
    print("   POST /api/analyze - Main analysis")
    print("   POST /api/models/compare - Model comparison")
    print("   GET  /api/demo/examples - Demo examples")
    print("   POST /api/batch - Batch processing")
    print("   GET  /api/health - Health check")
    print("   GET  /api/info - System info")
    print("="*80)

    # Run application
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True,
        use_reloader=False  # Avoid double initialization
    )