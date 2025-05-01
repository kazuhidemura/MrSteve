import torch
import numpy as np
import math
from dataclasses import dataclass
from collections import deque
from pdc_dp_means import MiniBatchDPMeans
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components


@dataclass
class ScoreStats:
    candidate_score: float | None
    mean_score: float
    stddev: float
    max_score: float

def merge_clusters(new_centers, new_labels, similarity_matrix, threshold):
    # Step 1: Threshold the similarity matrix
    adjacency_matrix = similarity_matrix > threshold
    
    # Step 2: Find connected components (clusters to merge)
    n_components, component_labels = connected_components(csgraph=adjacency_matrix, directed=False, connection='strong')
    
    # Step 3: Merge the cluster centers
    merged_centers = []
    for i in range(n_components):
        # Get all centers that belong to the same component
        cluster_indices = np.where(component_labels == i)[0]
        # Average these centers to create the new merged center
        merged_center = new_centers[cluster_indices].mean(axis=0)
        merged_centers.append(merged_center)
    
    merged_centers = np.vstack(merged_centers)  # Stack into numpy array
    
    # Step 4: Update the new_labels according to the new cluster assignment
    updated_labels = np.zeros_like(new_labels)
    for i, label in enumerate(component_labels):
        updated_labels[new_labels == i] = label
    
    return merged_centers, updated_labels


def calculate_score(mineclip, video_embeds, first_clip):
    video_feature = video_embeds
    if mineclip.reward_head.video_residual_weight is None:
        adapted_img = mineclip.reward_head.video_adapter(video_feature)
    else:
        res = torch.sigmoid(mineclip.reward_head.video_residual_weight)
        adapted_img = res * video_feature + (1.0 - res) * mineclip.reward_head.video_adapter(
            video_feature
        )
        
    if mineclip.reward_head.video_residual_weight is None:
        adapted_img_first = mineclip.reward_head.video_adapter(first_clip)
    else:
        res = torch.sigmoid(mineclip.reward_head.video_residual_weight)
        adapted_img_first = res * first_clip + (1.0 - res) * mineclip.reward_head.video_adapter(
            first_clip
        )
    return mineclip.reward_head.clip_model(adapted_img, adapted_img_first)[0].cpu().numpy()  # N
 

class PlaceEventMemory:
    def __init__(self, config):
        self.config = config
        self.cap = math.inf if config.cap is None else config.cap
        self.cluster_size = config.cluster_size
        self.cluster_yaw = config.cluster_yaw
        self.update_freq = config.update_freq
        self.cluster_freq = config.cluster_freq
        self.n_cluster_per_update = self.update_freq // self.cluster_freq
        self.thr = config.thr
        self.thr2 = config.thr2
        self.topk = config.topk

    def reset(self, init_log):
        self.clusters = []
        self.init_pos = init_log['init_pos']
        self.init_cam = init_log['init_cam']
    
    def __len__(self):
        mem_len = 0
        for cluster in self.clusters:
            for event_cluster in cluster['event_clusters']:
                mem_len += len(event_cluster['records'])
        return mem_len

    def add(self, record, mineclip, device):
        cls, clsh = self.cluster_size, self.cluster_size / 2.
        cly, clyh = self.cluster_yaw, self.cluster_yaw / 2.
        ts, emb, pos, cam = record.timestep, record.frame, record.pos, record.cam
        yaw = cam[1, 0] % 360
        
        poso = pos - self.init_pos
        yawo = (yaw + clyh) % 360 - clyh
        center_poso_ds = ((poso + clsh) // cls).astype(np.int32)
        center_yawo_ds = ((yaw + clyh) % 360 // cly).astype(np.int32)
        center_poso = center_poso_ds * cls 
        center_yawo = center_yawo_ds * cly
        
        # add to existing cluster
        added_to_cluster = False
        for cluster in self.clusters:
            if np.all(cluster['center_poso_ds'] == center_poso_ds) and np.all(cluster['center_yawo_ds'] == center_yawo_ds):
                cluster_size = int(np.sum([len(event_cluster['records']) for event_cluster in cluster['event_clusters']]))
                
                if cluster['is_clustered']:  
                    cluster['update'] += 1
                    cluster['dummy_buffer'].append(record)
                    if cluster['update'] >= self.update_freq:
                        X = list(cluster['dummy_buffer'])   
                        cluster_centers = cluster['event_clusters_centers']
                        feats = np.stack([record.frame for record in X], axis=0)
            
                        # Apply DP-means
                        dpmeans = MiniBatchDPMeans(n_clusters=self.n_cluster_per_update)
                        dpmeans = dpmeans.partial_fit(feats)
                        new_centers = dpmeans.cluster_centers_
                        new_labels = dpmeans.predict(feats)
                        scores = calculate_score(mineclip,
                                                 torch.from_numpy(new_centers).float().to(device),
                                                 torch.from_numpy(new_centers).float().to(device))
                        new_centers, new_labels = merge_clusters(new_centers, new_labels, scores, self.thr2)
                        new_labels = new_labels + len(cluster_centers)

                        # Merge new clusters with existing centers
                        offset = 0
                        for j in range(len(new_centers)):
                            cluster_id = j + len(cluster_centers) - offset
                            if np.sum(new_labels == cluster_id) == 0:
                                new_centers = np.delete(new_centers, j - offset, axis=0)
                                new_labels[new_labels > cluster_id] -= 1
                                offset += 1

                        scores = calculate_score(mineclip,
                                         torch.from_numpy(cluster_centers).float().to(device),
                                         torch.from_numpy(new_centers).float().to(device))
                        max_scores = np.max(scores, axis=0)
                        max_scores_idx = np.argmax(scores, axis=0)
                        is_removes = max_scores > self.thr 

                        offset = 0
                        for j in range(len(new_centers)):
                            cluster_id = j + len(cluster_centers) - offset
                            if is_removes[j]:
                                new_labels[new_labels == cluster_id] = max_scores_idx[j]
                                new_labels[new_labels > cluster_id] -= 1
                                offset += 1
                                
                        new_centers = new_centers[np.logical_not(is_removes)]
                        
                        for j in range(len(new_centers)):
                            cluster['event_clusters'].append({'center_emb': None, 
                                                              'records':[],
                                                              'label': j+len(cluster_centers),})
                        
                        # Add records to each cluster
                        for j, x in enumerate(X):
                            cluster['event_clusters'][new_labels[j]]['records'].append(x) 
                            
                            # If the memory is full, remove the most recent record in largest event cluster 
                            if len(self) >= self.cap:
                                len_list = np.array([len(cluster['records']) for cluster in cluster['event_clusters']])
                                max_length = np.max(len_list)

                                remove_candidates = np.where(len_list == max_length)[0]
                                remove_cluster_idx = remove_candidates[0]
                                for candidate in remove_candidates:
                                    cur_timestep = cluster["event_clusters"][remove_cluster_idx]["records"][0].timestep
                                    new_cand_timestep = cluster["event_clusters"][candidate]["records"][0].timestep

                                    if new_cand_timestep < cur_timestep:
                                        remove_cluster_idx = candidate

                                max_cluster = cluster['event_clusters'][remove_cluster_idx]
                                max_cluster['records'].pop(0)
                        
                        # Update cluster centers
                        for j in range(len(new_centers)):
                            cluster_records = cluster['event_clusters'][j+len(cluster_centers)]['records']
                            feats = np.stack([record.frame for record in cluster_records], axis=0)
                            dist = cdist(feats, new_centers[j:j+1])
                            center_emb = cluster_records[np.argmin(dist)].frame
                            cluster['event_clusters'][j+len(cluster_centers)]['center_emb'] = center_emb
                        
                        cluster['event_clusters_centers'] = np.concatenate([cluster_centers, np.copy(new_centers)]) 
                        cluster['update'] = 0

                        # Remove empty clusters
                        len_list = np.array([len(event_cluster['records']) for event_cluster in cluster['event_clusters']])
                        empty_clusters = np.where(len_list == 0)[0]
                        cluster['event_clusters'] = [event_cluster for i, event_cluster in enumerate(cluster['event_clusters']) if i not in empty_clusters]
                        cluster['event_clusters_centers'] = np.delete(cluster['event_clusters_centers'], empty_clusters, axis=0)
                else:
                    if cluster_size >= self.update_freq:
                        
                        cluster['event_clusters_centers'] = np.empty((0, 512))
                        records_0 = cluster['event_clusters'][0]['records']
                        cluster['event_clusters'].pop(0)    
                        feats = np.stack([record.frame for record in records_0], axis=0)  # N,512

                        dpmeans = MiniBatchDPMeans(n_clusters=self.n_cluster_per_update)
                        dpmeans = dpmeans.partial_fit(feats)
                        new_centers = dpmeans.cluster_centers_
                        new_labels = dpmeans.predict(feats) + len(cluster['event_clusters'])
                        scores = calculate_score(mineclip,
                                                 torch.from_numpy(new_centers).float().to(device),
                                                 torch.from_numpy(new_centers).float().to(device))  # len(cluster_centers),len(new_centers)
                        new_centers, new_labels = merge_clusters(new_centers, new_labels, scores, self.thr2)

                        offset = 0
                        for j in range(len(new_centers)):
                            cluster_id = j + len(cluster['event_clusters']) - offset
                            if np.sum(new_labels == cluster_id) == 0:
                                new_centers = np.delete(new_centers, j - offset, axis=0)
                                new_labels[new_labels > cluster_id] -= 1
                                offset += 1

                        for j in range(len(new_centers)):
                            cluster['event_clusters'].append({'center_emb': None,
                                                              'records':[],
                                                              'label': j,}) 
                        
                        for j, x in enumerate(records_0):
                            cluster['event_clusters'][new_labels[j]]['records'].append(x) 
                        
                        for j in range(len(new_centers)):
                            cluster_records = cluster['event_clusters'][j]['records']
                            feats = np.stack([record.frame for record in cluster_records], axis=0)  # N,512
                            dist = cdist(feats, new_centers[j:j+1])
                            closest_record = cluster_records[np.argmin(dist)]
                            cluster['event_clusters'][j]['center_emb'] = closest_record.frame
                        
                        cluster['event_clusters_centers'] = np.concatenate([cluster['event_clusters_centers'], np.copy(new_centers)])
                        cluster['is_clustered'] = True
                    else:
                        cluster['event_clusters'][0]['records'].append(record)
                
                center_poso_dist = np.linalg.norm(poso - center_poso)
                center_yawo_dist = np.linalg.norm(yawo - center_yawo)
                
                if center_poso_dist <= cluster['center_poso_dist'] and center_yawo_dist <= cluster['center_yawo_dist']:
                    cluster['center_poso_dist'] = center_poso_dist
                    cluster['center_yawo_dist'] = center_yawo_dist  
                    cluster['center_pos'] = pos
                    cluster['center_yaw'] = yaw
                    cluster['center_emb'] = emb
                    
                added_to_cluster = True
                break

        # create new cluster
        if not added_to_cluster:
            self.clusters.append({
                'center_poso_ds': center_poso_ds,
                'center_yawo_ds': center_yawo_ds,
                'center_poso_dist': np.linalg.norm(poso - center_poso),
                'center_yawo_dist': np.linalg.norm(yawo - center_yawo),
                'center_pos': pos,
                'center_yaw': yaw,
                'center_emb': emb,
                'is_clustered': False,
                'event_clusters_centers': None,
                'update': 0,
                'dummy_buffer': deque(maxlen=self.update_freq),
                'event_clusters': [{'center_emb': emb,
                                    'records': [record],
                                    'label': -1}],
            })

        if len(self) >= self.cap:
            
            scluster_size = []
            for cluster in self.clusters:
                cluster_size = int(np.sum([len(event_cluster['records']) for event_cluster in cluster['event_clusters']]))
                scluster_size.append(cluster_size)
            
            max_scluster_idx = np.argmax(scluster_size)
            max_scluster = self.clusters[max_scluster_idx]
            
            ecluster_size = []
            for event_cluster in max_scluster['event_clusters']:
                ecluster_size.append(len(event_cluster['records']))
            
            max_ecluster_idx = np.argmax(ecluster_size)
            max_scluster['event_clusters'][max_ecluster_idx]['records'].pop(0)
              
    def query(self, current_pos, text_embeds, mineclip, clip_threshold):
        idx_center_embs = []
        for j, cluster in enumerate(self.clusters):
            for i, event_cluster in enumerate(cluster['event_clusters']):
                idx_center_embs.append((j, i, event_cluster['center_emb']))
        
        video_embeds = torch.from_numpy(np.stack([
            idx_center_emb[2] for idx_center_emb in idx_center_embs
        ], axis=0)).float().to(text_embeds.device)
        
        # caculate scores and pick top k clusters
        scores = mineclip.reward_head(video_embeds, text_embeds)[0].cpu().numpy()

        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        scores_max = np.max(scores)

        topk_idx = np.argsort(scores[:, 0])[::-1]
        if self.topk < len(topk_idx):
            topk_idx = topk_idx[:self.topk]
        
        topk_clusters = []
        for idx in topk_idx:
            scluster_idx = idx_center_embs[idx][0]
            ecluster_idx = idx_center_embs[idx][1]
            topk_clusters.append(self.clusters[scluster_idx]['event_clusters'][ecluster_idx])
        
        # extract top-k and its records embedding 
        topk_records = []
        for cluster in topk_clusters:
            for record in cluster['records']:
                topk_records.append(record)
        
        # calculate score pick the best
        video_embeds = torch.from_numpy(np.stack([
            record.frame for record in topk_records
        ], axis=0)).float().to(text_embeds.device)
        scores = mineclip.reward_head(video_embeds, text_embeds)[0].cpu().numpy()

        candidates_idx = np.where(scores >= clip_threshold)[0]
        candidate = None
        candidate_score = None
        distance = np.inf
        for candidate_idx in candidates_idx:
            score = scores[candidate_idx]
            record = topk_records[candidate_idx]
            d = np.linalg.norm(record.pos - current_pos)
            if d < distance:
                distance = d
                candidate = record
                candidate_score = score
        return candidate, ScoreStats(
            candidate_score=candidate_score,
            mean_score=scores_mean,
            stddev=scores_std,
            max_score=scores_max,
        )
    
    
    def get_status(self):
        num_sclusters = len(self.clusters)
        event_cluster_lens = []
        for cluster in self.clusters:
            for event_cluster in cluster['event_clusters']:
                event_cluster_lens.append(len(event_cluster['records']))
        
        num_eclusters = len(event_cluster_lens)
        cluster_len_max = int(np.max(event_cluster_lens))  # cluster with maximum length
        cluster_len_min = int(np.min(event_cluster_lens))  # cluster with minimum length
        cluster_len_avg = int(np.mean(event_cluster_lens))  # average cluster length
        buffer_len = len(self)
        
        return {'buffer_len': buffer_len,
                'num_sclusters': num_sclusters,
                'num_eclusters': num_eclusters,
                'cluster_len_max': cluster_len_max,
                'cluster_len_min': cluster_len_min,
                'cluster_len_avg': cluster_len_avg}


class EventMemory:
    def __init__(self, config):
        self.config = config
        self.cap = math.inf if config.cap is None else config.cap
        self.update_freq = config.update_freq
        self.cluster_freq = config.cluster_freq
        self.n_cluster_per_update = self.update_freq // self.cluster_freq
        self.thr = config.thr
        self.thr2 = config.thr2
        self.topk = config.topk
    
    def reset(self, init_log):
        self.clusters = []
        self.embed_buffer = deque(maxlen=self.update_freq)
        self.update = 0
        self.cluster_centers = np.empty((0, 512))
    
    def __len__(self):
        return int(np.sum([len(cluster['records']) for cluster in self.clusters]))
    
    def add(self, record, mineclip, device):
        self.update += 1
        self.embed_buffer.append(record)
        
        # update the memory every update_freq
        if self.update >= self.update_freq:
            
            X = list(self.embed_buffer)
            feats = np.stack([record.frame for record in X], axis=0)
            
            # apply DP-means
            dpmeans = MiniBatchDPMeans(n_clusters=self.n_cluster_per_update)
            dpmeans = dpmeans.partial_fit(feats)
            new_centers = dpmeans.cluster_centers_
            new_labels = dpmeans.predict(feats)
            scores = calculate_score(mineclip,
                                     torch.from_numpy(new_centers).float().to(device),
                                     torch.from_numpy(new_centers).float().to(device))
            
            new_centers, new_labels = merge_clusters(new_centers, new_labels, scores, self.thr2)
            new_labels = new_labels + len(self.cluster_centers)
            
            offset = 0
            for j in range(len(new_centers)):
                cluster_id = j + len(self.cluster_centers) - offset
                if np.sum(new_labels == cluster_id) == 0:
                    new_centers = np.delete(new_centers, j - offset, axis=0)
                    new_labels[new_labels > cluster_id] -= 1
                    offset += 1
            
            if len(self.cluster_centers) > 0:
                scores = calculate_score(mineclip,
                                         torch.from_numpy(self.cluster_centers).float().to(device),
                                         torch.from_numpy(new_centers).float().to(device))
                max_scores = np.max(scores, axis=0)
                max_scores_idx = np.argmax(scores, axis=0)
                is_removes = max_scores > self.thr 
            
                orig_new_centers_cnt = len(new_centers)
                new_labels_copy = new_labels.copy()

                offset = 0
                for j in range(len(new_centers)):
                    cluster_id = j + len(self.cluster_centers) - offset
                    if is_removes[j]:
                        new_labels[new_labels == cluster_id] = max_scores_idx[j]
                        new_labels[new_labels > cluster_id] -= 1
                        offset += 1
                        
                new_centers = new_centers[np.logical_not(is_removes)]
                
                # make initial cluster
                for j in range(len(new_centers)):
                    self.clusters.append({'center_emb': None, 
                                          'records':[],
                                          'label': j+len(self.cluster_centers),})
                
                for j, x in enumerate(X):
                    self.clusters[new_labels[j]]['records'].append(x) 
                    
                    # if out-of-memory, remove the most recent record in largest event cluster 
                    if len(self) >= self.cap:
                        len_list = np.array([len(cluster['records']) for cluster in self.clusters])
                        max_length = np.max(len_list)

                        remove_candidates = np.where(len_list == max_length)[0]
                        remove_cluster_idx = remove_candidates[0]
                        for candidate in remove_candidates:
                            cur_timestep = self.clusters[remove_cluster_idx]["records"][0].timestep
                            new_cand_timestep = self.clusters[candidate]["records"][0].timestep

                            if new_cand_timestep < cur_timestep:
                                remove_cluster_idx = candidate

                        max_cluster = self.clusters[remove_cluster_idx]
                        max_cluster['records'].pop(0)
 
                # update cluster centers
                for j in range(len(new_centers)):
                    cluster_records = self.clusters[j+len(self.cluster_centers)]['records']
                    assert len(cluster_records) > 0, 'strange behavior observed in add function in EventFIFOMemory'
                    
                    feats = np.stack([record.frame for record in cluster_records], axis=0)
                    dist = cdist(feats, new_centers[j:j+1])
                    center_emb = cluster_records[np.argmin(dist)].frame
                    self.clusters[j+len(self.cluster_centers)]['center_emb'] = center_emb
                
                self.cluster_centers = np.concatenate([self.cluster_centers, np.copy(new_centers)])

                # remove empty clusters
                len_list = np.array([len(cluster['records']) for cluster in self.clusters])
                empty_clusters = np.where(len_list == 0)[0]
                self.clusters = [cluster for i, cluster in enumerate(self.clusters) if i not in empty_clusters]
                self.cluster_centers = np.delete(self.cluster_centers, empty_clusters, axis=0)
            else:
                for j in range(len(new_centers)):
                    self.clusters.append({'center_emb': None,
                                          'records':[],
                                          'label': j,}) 
                
                for j, x in enumerate(X):
                    self.clusters[new_labels[j]]['records'].append(x) 
                
                for j in range(len(new_centers)):
                    cluster_records = self.clusters[j]['records']
                    feats = np.stack([record.frame for record in cluster_records], axis=0)
                    dist = cdist(feats, new_centers[j:j+1])
                    closest_record = cluster_records[np.argmin(dist)]
                    self.clusters[j]['center_emb'] = closest_record.frame
            
                self.cluster_centers = np.concatenate([self.cluster_centers, np.copy(new_centers)])
            
            self.update = 0
        
    def query(self, current_pos, text_embeds, mineclip, clip_threshold):
        video_embeds = torch.from_numpy(np.stack([
            cluster['center_emb'] for cluster in self.clusters
        ], axis=0)).float().to(text_embeds.device)
        
        # caculate scores and pick top k clusters
        scores = mineclip.reward_head(video_embeds, text_embeds)[0].cpu().numpy()

        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        scores_max = np.max(scores)

        topk_idx = np.argsort(scores[:, 0])[::-1]
        if self.topk < len(topk_idx):
            topk_idx = topk_idx[:self.topk]
        
        topk_clusters = [self.clusters[idx] for idx in topk_idx]
        
        topk_records = []
        for cluster in topk_clusters:
            for record in cluster['records']:
                topk_records.append(record)
        
        # calculate score pick the best
        video_embeds = torch.from_numpy(np.stack([
            record.frame for record in topk_records
        ], axis=0)).float().to(text_embeds.device)
        scores = mineclip.reward_head(video_embeds, text_embeds)[0].cpu().numpy()

        candidates_idx = np.where(scores >= clip_threshold)[0]
        candidate = None
        candidate_score = None
        distance = np.inf
        for candidate_idx in candidates_idx:
            score = scores[candidate_idx]
            record = topk_records[candidate_idx]
            d = np.linalg.norm(record.pos - current_pos)
            if d < distance:
                distance = d
                candidate = record
                candidate_score = score
        return candidate, ScoreStats(
            candidate_score=candidate_score,
            mean_score=scores_mean,
            stddev=scores_std,
            max_score=scores_max,
        )

    def get_status(self):        
        num_clusters = len(self.clusters)
        cluster_lens = [len(cluster['records']) for cluster in self.clusters]
        cluster_len_max = int(np.max(cluster_lens))  # cluster with maximum length
        cluster_len_min = int(np.min(cluster_lens))  # cluster with minimum length
        cluster_len_avg = int(np.mean(cluster_lens))  # average cluster length
        buffer_len = int(np.sum(cluster_lens))  # memory length
        
        return {'topk': self.topk,
                'thr': self.thr,
                'buffer_len': buffer_len,
                'num_clusters': num_clusters,
                'cluster_len_max': cluster_len_max,
                'cluster_len_min': cluster_len_min,
                'cluster_len_avg': cluster_len_avg}


class PlaceMemory:
    def __init__(self, config):
        self.config = config
        self.cap = math.inf if config.cap is None else config.cap
        self.cluster_size = config.cluster_size
        self.cluster_yaw = config.cluster_yaw
        self.topk = config.topk

    def reset(self, init_log):
        self.clusters = []
        self.init_pos = init_log['init_pos']
        self.init_cam = init_log['init_cam']
    
    def __len__(self):
        return int(np.sum([len(cluster['records']) for cluster in self.clusters]))

    def add(self, record, mineclip, device):
        cls, clsh = self.cluster_size, self.cluster_size / 2.
        cly, clyh = self.cluster_yaw, self.cluster_yaw / 2.
        ts, emb, pos, cam = record.timestep, record.frame, record.pos, record.cam
        yaw = cam[1, 0] % 360
        
        poso = pos - self.init_pos
        yawo = (yaw + clyh) % 360 - clyh
        center_poso_ds = ((poso + clsh) // cls).astype(np.int32)
        center_yawo_ds = ((yaw + clyh) % 360 // cly).astype(np.int32)
        center_poso = center_poso_ds * cls
        center_yawo = center_yawo_ds * cly
        
        # add to existing cluster
        added_to_cluster = False
        for cluster in self.clusters:
            if np.all(cluster['center_poso_ds'] == center_poso_ds) and np.all(cluster['center_yawo_ds'] == center_yawo_ds):
                
                # add to existing cluster
                cluster['records'].append(record)
                center_poso_dist = np.linalg.norm(poso - center_poso)
                center_yawo_dist = np.linalg.norm(yawo - center_yawo)
                
                if center_poso_dist <= cluster['center_poso_dist'] and center_yawo_dist <= cluster['center_yawo_dist']:
                    cluster['center_poso_dist'] = center_poso_dist
                    cluster['center_yawo_dist'] = center_yawo_dist  
                    cluster['center_pos'] = pos
                    cluster['center_yaw'] = yaw
                    cluster['center_emb'] = emb
                    
                added_to_cluster = True
                break

        # create new cluster
        if not added_to_cluster:
            self.clusters.append({
                'center_poso_ds': center_poso_ds,
                'center_yawo_ds': center_yawo_ds,
                'center_poso_dist': np.linalg.norm(poso - center_poso),
                'center_yawo_dist': np.linalg.norm(yawo - center_yawo),
                'center_pos': pos,
                'center_yaw': yaw,
                'center_emb': emb,
                'records': [record]
            })

        # if out-of-memory, remove the most recent record in largest spatial cluster 
        if len(self) >= self.cap:
            len_list = np.array([len(cluster['records']) for cluster in self.clusters])
            max_length = np.max(len_list)

            remove_candidates = np.where(len_list == max_length)[0]
            remove_cluster_idx = remove_candidates[0]
            for candidate in remove_candidates:
                cur_timestep = self.clusters[remove_cluster_idx]["records"][0].timestep
                new_cand_timestep = self.clusters[candidate]["records"][0].timestep

                if new_cand_timestep < cur_timestep:
                    remove_cluster_idx = candidate

            max_cluster = self.clusters[remove_cluster_idx]
            max_cluster['records'].pop(0)
            if len(max_cluster['records']) == 0:
                del self.clusters[remove_cluster_idx]
        
    def query(self, current_pos, text_embeds, mineclip, clip_threshold):
        video_embeds = torch.from_numpy(np.stack([
            cluster['center_emb'] for cluster in self.clusters
        ], axis=0)).float().to(text_embeds.device)
        
        scores = mineclip.reward_head(video_embeds, text_embeds)[0].cpu().numpy()

        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        scores_max = np.max(scores)

        topk_idx = np.argsort(scores[:, 0])[::-1]
        if self.topk < len(topk_idx):
            topk_idx = topk_idx[:self.topk]

        topk_clusters = [self.clusters[idx] for idx in topk_idx]
        
        topk_records = []
        for cluster in topk_clusters:
            for record in cluster['records']:
                topk_records.append(record)
        
        # calculate score pick the best
        video_embeds = torch.from_numpy(np.stack([
            record.frame for record in topk_records
        ], axis=0)).float().to(text_embeds.device)
        scores = mineclip.reward_head(video_embeds, text_embeds)[0].cpu().numpy()

        candidates_idx = np.where(scores >= clip_threshold)[0]
        candidate = None
        candidate_score = None
        distance = np.inf
        for candidate_idx in candidates_idx:
            score = scores[candidate_idx]
            record = topk_records[candidate_idx]
            d = np.linalg.norm(record.pos - current_pos)
            if d < distance:
                distance = d
                candidate = record
                candidate_score = score
        return candidate, ScoreStats(
            candidate_score=candidate_score,
            mean_score=scores_mean,
            stddev=scores_std,
            max_score=scores_max,
        )
    
    def get_status(self):
        num_clusters = len(self.clusters)
        cluster_lens = [len(cluster['records']) for cluster in self.clusters]
        cluster_len_max = int(np.max(cluster_lens))  # cluster with maximum length
        cluster_len_min = int(np.min(cluster_lens))  # cluster with minimum length
        cluster_len_avg = int(np.mean(cluster_lens))  # average cluster length
        buffer_len = int(np.sum(cluster_lens))  # memory length
        
        return {'buffer_len': buffer_len,
                'num_clusters': num_clusters,
                'cluster_len_max': cluster_len_max,
                'cluster_len_min': cluster_len_min,
                'cluster_len_avg': cluster_len_avg}


class FIFOMemory:
    def __init__(self, config):
        self.config = config
        self.cap = math.inf if config.cap is None else config.cap
    
    def reset(self, init_log):
        self.clusters = []
        self.embed_buffer = []
    
    def __len__(self):
        return len(self.embed_buffer)
    
    def add(self, record, mineclip, device):
        self.embed_buffer.append(record)
        if len(self) >= self.cap:
            self.embed_buffer.pop(0)
        
    def query(self, current_pos, text_embeds, mineclip, clip_threshold):
        video_embeds = torch.from_numpy(np.stack([
            record.frame for record in self.embed_buffer
        ], axis=0)).float().to(text_embeds.device)
        
        scores = mineclip.reward_head(video_embeds, text_embeds)[0].cpu().numpy()

        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        scores_max = np.max(scores)

        candidates_idx = np.where(scores >= clip_threshold)[0]
        candidate = None
        candidate_score = None
        distance = np.inf
        for candidate_idx in candidates_idx:
            score = scores[candidate_idx]
            record = self.embed_buffer[candidate_idx]
            d = np.linalg.norm(record.pos - current_pos)
            if d < distance:
                distance = d
                candidate = record
                candidate_score = score
        
        return candidate, ScoreStats(
            candidate_score=candidate_score,
            mean_score=scores_mean,
            stddev=scores_std,
            max_score=scores_max,
        )

    def get_status(self):
        return {'buffer_len': len(self.embed_buffer)}
        

MEMORY_CLS = {
    "place_event_memory": PlaceEventMemory,
    "event_memory": EventMemory,
    "place_memory": PlaceMemory,
    "fifo_memory": FIFOMemory,
}
