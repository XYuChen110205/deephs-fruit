"""TDD 测试 - HS-CNN 高光谱检测系统 API 测试"""

import json
import os
import sys
import unittest
from pathlib import Path

# Add the templates directory to path
TEMPLATES_DIR = Path(__file__).parent.parent / "HSI_Fruit_Maturity_Detection" / "HSI_Fruit_Maturity_Detection" / "templates"
sys.path.insert(0, str(TEMPLATES_DIR))

from app import app


class TestAPI(unittest.TestCase):
    """API 端点测试"""

    def setUp(self):
        self.client = app.test_client()

    def test_status_endpoint(self):
        """测试 /status 返回正确的系统状态"""
        resp = self.client.get('/status')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('model_loaded', data)
        self.assertIn('device', data)
        self.assertIn('model_path', data)

    def test_ablation_api(self):
        """测试 /api/ablation 返回消融实验数据"""
        resp = self.client.get('/api/ablation')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('dataset', data)
        self.assertIn('ablation_results', data)
        self.assertGreater(len(data['ablation_results']), 0)
        for result in data['ablation_results']:
            self.assertIn('name', result)
            self.assertIn('sample_acc', result)
            self.assertIn('fruit_acc', result)

    def test_comparison_api(self):
        """测试 /api/comparison 返回对比实验数据"""
        resp = self.client.get('/api/comparison')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('dataset', data)
        self.assertIn('comparison_results', data)
        self.assertGreater(len(data['comparison_results']), 0)
        for result in data['comparison_results']:
            self.assertIn('name', result)
            self.assertIn('sample_acc', result)
            self.assertIn('f1_score', result)

    def test_home_page(self):
        """测试首页返回 HTML"""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/html', resp.content_type)


if __name__ == '__main__':
    unittest.main(verbosity=2)
