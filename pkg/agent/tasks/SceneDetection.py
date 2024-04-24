import json
import os

import requests

from .AbstractTask import AbstractTask, TaskNames

from pkg.agent.tasks.lib import scenedetector


VIDEO_SCENEDATA_KEY = 'sceneData'


class SceneDetection(AbstractTask):

    @staticmethod
    def get_name():
        return TaskNames.SceneDetection

    @staticmethod
    def get_file_path(video):
        return video['video1']['path']
        # legacy: return os.path.join(DATA_DIRECTORY, '%s.mp4' % video['video1']['id'])

    def find_scenes(self, video_id, video, readonly):
        # get file_path from video
        # Note that because we're accessing the raw file, we're assuming that
        # we're running on the same server and/or in the same file space
        # TODO: process multiple videos?
        file_path = self.get_file_path(video=video)

        # Call scenedetector.find_scenes, store scene data back in api
        try:
            self.logger.info(' [%s] SceneDetection getting scenes for %s...' % (video_id, file_path))
            scenes, scenes_meta = scenedetector.find_scenes(video_path=file_path)

            # save found scenes to video in api
            #self.logger.debug(' [%s] SceneDetection found scenes: %s' % (video_id, scenes))
            if readonly:
                self.logger.info(' [%s] SceneDetection running as READONLY.. scenes have not been saved' % (video_id))
            else:
                self.jwt = self.update_jwt()
                resp = requests.post(url='%s/api/Task/UpdateSceneData?videoId=%s' % (self.target_host, video_id),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % self.jwt},
                                     data=json.dumps({"Scenes": scenes, "ScenesMetadata": scenes_meta}))

                resp.raise_for_status()
                #self.logger.debug(' [%s] SceneDetection successfully saved scenes: %s' % (video_id, scenes))

            return scenes
        except Exception as e:
            self.logger.error(
                ' [%s] SceneDetection failed to detect scenes in videoId=%s: %s' % (
                    video_id, file_path, str(e)))
            return None

    # Message Body Format:
    #     {'Data': 'db2090f7-09f2-459a-84b9-96bd2f506f68',
    #     'TaskParameters': {'Force': False, 'Metadata': None}}
    def run_task(self, body, emitter):
        self.logger.info(' [.] SceneDetection message recv\'d: %s' % body)
        video_id = body['Data']
        parameters = body.get('TaskParameters', {})
        force = parameters.get('Force', False)
        readonly = parameters.get('ReadOnly', False)
        self.logger.info(' [%s] SceneDetection started on videoId=%s...' % (video_id, video_id))

        # fetch video metadata by id to get path
        video = self.get_video(video_id=video_id)

        # short-circuit if we already have scene data
        if not force and VIDEO_SCENEDATA_KEY in video and video[VIDEO_SCENEDATA_KEY]:
            self.logger.warning(' [%s] Skipping SceneDetection: sceneData already exists' % video_id)
            self.logger.info(' [%s] SceneDetection now triggering: PhraseHinter' % video_id)
            emitter.publish(routing_key='PhraseHinter', body=body)
            return

        # get file_path from video
        # Note that because we're accessing the raw file, we're assuming that
        # we're running on the same server and/or in the same file space
        # TODO: process multiple videos?
        file_path = self.get_file_path(video=video)

        # Short-circuit if we can't find the file
        # fetch the file from server if not found (if DOWNLOAD_MISSING_VIDEOS=True)
        if not self.ensure_file_exists(video_id=video_id, file_path=file_path):
            self.logger.warning(' [%s] Skipping SceneDetection: video file not found locally' % video_id)
            return

        # verify file size
        if 'fileMediaInfo' in video and 'format' in video['fileMediaInfo'] and 'size' in video['fileMediaInfo']['format']:
            expected_size = video['fileMediaInfo']['format']['size']
            actual_size = os.path.getsize(file_path)
            if int(actual_size) != int(expected_size):
                self.logger.warning('Size mismatch on downloaded file: %s (%s bytes, but should be %s)' %
                                    (video_id, actual_size, expected_size))

        self.find_scenes(video_id, video, readonly)

        if video is None:
            self.logger.error(' [%s] SceneDetection FAILED to lookup videoId=%s' % (video_id, video_id))
            return

        self.logger.info(' [%s] SceneDetection complete!' % video_id)

        # Trigger TranscriptionTask (which will generate captions in various languages)
        self.logger.info(' [%s] SceneDetection now triggering: FlashDetection' % video_id)
        #body['Scenes'] = scenes  # json.dumps(scenes)
        emitter.publish(routing_key='FlashDetection', body=body)

        return


